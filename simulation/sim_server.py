import os
import sys
import json
import asyncio
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

# Add project root to Python module search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sim_env_wrapper import SimEnvWrapper
from simulation.scenario_detector import classify_from_grid

app = FastAPI(title="UAV SAR Browser 2D Simulation Server")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_dir, exist_ok=True)

# Mission State Storage
class Mission:
    def __init__(self, mission_id: str, scenario: str, victims: list):
        self.mission_id = mission_id
        self.scenario = scenario
        self.victims = victims # List of dicts: {"x": c, "y": r}
        self.wrapper = None
        self.is_running = False
        self.is_paused = False
        self.task = None
        self.connected_clients: List[WebSocket] = []
        self.victim_states = [{"x": v["x"], "y": v["y"], "found": False, "found_at_tick": None} for v in victims]

missions: Dict[str, Mission] = {}

class VictimData(BaseModel):
    x: int # Col
    y: int # Row

class StartMissionRequest(BaseModel):
    scenario: str
    victims: List[VictimData]

@app.post("/api/start_mission")
async def start_mission(req: StartMissionRequest):
    mission_id = str(uuid.uuid4())
    victims_list = [{"x": v.x, "y": v.y} for v in req.victims]
    
    # Store mission pending connection
    missions[mission_id] = Mission(mission_id, req.scenario, victims_list)
    
    return {"mission_id": mission_id, "status": "ready"}

async def run_simulation_loop(mission: Mission):
    """Runs the pure Python simulation steps at 5Hz and broadcasts updates."""
    print(f"[Mission {mission.mission_id}] Starting simulation loop...", flush=True)
    
    # Convert [{"x": c, "y": r}] to list of [r, c] for the wrapper
    victims_2d = [[v["y"], v["x"]] for v in mission.victims]
    
    try:
        loop = asyncio.get_running_loop()
        # Offload the heavy PyTorch initialization to a worker thread to prevent asyncio deadlocks!
        print(f"[Mission {mission.mission_id}] Spawning wrapper in background thread...", flush=True)
        mission.wrapper = await loop.run_in_executor(None, SimEnvWrapper, mission.scenario, victims_2d)
    except Exception as e:
        print(f"[Mission {mission.mission_id}] Failed to init wrapper: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    mission.is_running = True
    
    try:
        # Step pacing of 200ms per step (5Hz)
        max_steps = 500
        
        for step_idx in range(max_steps):
            while mission.is_paused:
                await asyncio.sleep(0.1)
                
            if not mission.is_running:
                break
                
            # 1. Execute one step of the physical and policy logic in a worker thread!
            state_update = await loop.run_in_executor(None, mission.wrapper.step)
            tick = state_update["step_count"]
            
            # 2. Extract state for frontend JSON spec
            drones = []
            for i in range(4):
                pos = state_update["agent_positions"][i] # (row, col) => y, x
                r, c = pos
                
                # Check scenario detector logic
                # Pass local obs with the step count and actual scenario
                local_obs = {
                    "step_count": tick,
                    "scenario": mission.wrapper.scenario
                }
                detected_scenario = classify_from_grid(local_obs, mission.scenario)
                scenario_known = detected_scenario != "unknown"
                
                drones.append({
                    "id": i,
                    "x": int(c),
                    "y": int(r),
                    "scenario_known": bool(scenario_known),
                    "scenario": str(detected_scenario)
                })
                
            # Check for newly found victims
            victims_found_array = state_update["victim_found"]
            found_count = 0
            for idx, found in enumerate(victims_found_array):
                if found and not mission.victim_states[idx]["found"]:
                    mission.victim_states[idx]["found"] = True
                    mission.victim_states[idx]["found_at_tick"] = int(tick)
                if mission.victim_states[idx]["found"]:
                    found_count += 1
                    
            # Compute coverage
            coverage_grid = state_update["coverage_map"]
            obstacle_grid = state_update["obstacle_map"]
            
            total_cells = coverage_grid.size
            obstacle_cells = int((obstacle_grid == 1.0).sum())
            navigable_cells = max(1, total_cells - obstacle_cells)
            explored_cells = int((coverage_grid[obstacle_grid < 1.0] > 0).sum())
            
            coverage_pct = int((explored_cells / navigable_cells) * 100)
            
            msg = {
                "tick": int(tick),
                "drones": drones,
                "victims": mission.victim_states,
                "coverage": coverage_pct,
                "stats": {
                    "total": total_cells,
                    "obstacles": obstacle_cells,
                    "navigable": navigable_cells,
                    "covered": explored_cells,
                    "found": int(found_count),
                    "steps": int(tick)
                },
                "obstacles": obstacle_grid.tolist() if int(tick) == 1 else None
            }
            
            # Broadcast to all connected clients for this mission
            dead_clients = []
            for ws in mission.connected_clients:
                try:
                    await ws.send_json(msg)
                except Exception as e:
                    print(f"[Mission {mission.mission_id}] Error sending JSON to WS: {e}", flush=True)
                    dead_clients.append(ws)
            for ws in dead_clients:
                mission.connected_clients.remove(ws)
            
            # Stop if all victims are found
            if found_count == len(mission.victims) or state_update["term"] or state_update["trunc"]:
                print(f"[Mission {mission.mission_id}] Mission complete or truncated.", flush=True)
                break
                
            await asyncio.sleep(0.2) # 5Hz tick rate

            
    except asyncio.CancelledError:
        print(f"[Mission {mission.mission_id}] Task cancelled.", flush=True)
    except Exception as e:
        print(f"[Mission {mission.mission_id}] Error in loop: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        mission.is_running = False
        print(f"[Mission {mission.mission_id}] Stopped.", flush=True)


@app.websocket("/ws/mission/{mission_id}")
async def websocket_endpoint(websocket: WebSocket, mission_id: str):
    await websocket.accept()
    
    if mission_id not in missions:
        await websocket.send_json({"error": "Invalid mission ID"})
        await websocket.close()
        return
        
    mission = missions[mission_id]
    mission.connected_clients.append(websocket)
    
    # Start the simulation loop if it's not already running
    if not mission.is_running and mission.task is None:
        mission.task = asyncio.create_task(run_simulation_loop(mission))
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")
            
            if action == "pause":
                mission.is_paused = True
            elif action == "resume":
                mission.is_paused = False
                
    except WebSocketDisconnect:
        if websocket in mission.connected_clients:
            mission.connected_clients.remove(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        if websocket in mission.connected_clients:
            mission.connected_clients.remove(websocket)

# Fallback route to serve frontend index.html at root
@app.get("/")
def get_root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("Frontend files not found. Please create simulation/frontend/index.html")

# Serve other static files (js, css, images) from frontend folder
app.mount("/", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    print("[UAV SAR] Starting pure Python 2D simulation server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
