import os
import sys
import time
import numpy as np

# Add parent directory and uav_sar directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uav_sar")))

from uav_sar.airsim_wrapper import AirSimWrapper, HAS_AIRSIM
from uav_sar.environment.disaster_env import DisasterEnv
from simulation.mission_config import MissionConfig
from simulation.scenario_detector import ScenarioDetector
from simulation.victim_tracker import VictimTracker

class AirSimEnvAdapter(AirSimWrapper):
    """
    Subclasses AirSimWrapper to add landing, sensor-based scenario detection,
    coordinate mapping, and WebSocket state integration.
    """
    def __init__(self, checkpoint_path=None, config_dir=None):
        # Resolve config and checkpoint paths in uav_sar directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if config_dir is None:
            config_dir = os.path.join(self.project_root, "uav_sar", "config")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.project_root, "uav_sar", "checkpoints", "latest.pt")
            
        super().__init__(checkpoint_path=checkpoint_path, config_dir=config_dir)
        
        # Load simulation config
        self.mission_config = MissionConfig()
        
        # Overwrite default victim positions with custom operator placements
        if self.mission_config.victim_positions_3d:
            self.victim_positions_3d = self.mission_config.victim_positions_3d
            print(f"[AirSimEnvAdapter] Loaded {len(self.victim_positions_3d)} user-placed victims.")
        else:
            print("[AirSimEnvAdapter] No user-placed victims found. Using wrapper defaults.")
            
        self.victim_found = np.zeros(len(self.victim_positions_3d), dtype=bool)
        
        # Initialize sub-components
        self.detector = ScenarioDetector()
        self.tracker = VictimTracker(
            grid_size=self.grid_size,
            cell_size_meters=self.cell_size_meters,
            lat_ref=self.mission_config.lat_ref,
            lon_ref=self.mission_config.lon_ref
        )
        
        self.detected_scenario = "unknown"
        self.drones_landed = False
        
        # Track coordinates of drone paths for rendering in frontend
        self.drone_trails = {f"drone_{i}": [] for i in range(4)}

    def run_landing_and_detection(self):
        """
        Executes the landing phase. Drones take off/spawn, land on the map,
        collect sensor observations, run scenario classification, and initialize the environment.
        """
        print("\n=== STARTING LANDING & SCENARIO DETECTION PHASE ===")
        
        if HAS_AIRSIM and not self.is_mock:
            # 1. Spawn / Takeoff in AirSim
            print("[AirSimEnvAdapter] Spawning drones in AirSim...")
            drone_names = [("SimpleFlight" if i==0 else f"drone_{i}") for i in range(4)]
            for name in drone_names:
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
            
            takeoff_tasks = [self.client.takeoffAsync(vehicle_name=name) for name in drone_names]
            for t in takeoff_tasks:
                t.join()
                
            # Hover at landing height (e.g. 2 meters above terrain)
            print("[AirSimEnvAdapter] Drones descending to landing positions...")
            land_tasks = [self.client.moveToZAsync(-2.0, 2.0, vehicle_name=name) for name in drone_names]
            for t in land_tasks:
                t.join()
                
            # Collect real sensor data from first drone
            print("[AirSimEnvAdapter] Analyzing real AirSim sensor signals...")
            try:
                # 1. Smoke (Camera RGB image)
                import airsim
                responses = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ], vehicle_name="SimpleFlight")
                
                rgb_smoke = 0.0
                if responses:
                    rgb_smoke = self.detector.extract_smoke_from_image(responses[0].image_data_uint8)
                    
                # 2. LiDAR density
                lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="SimpleFlight")
                points = np.array(lidar_data.point_cloud).reshape(-1, 3) if len(lidar_data.point_cloud) >= 3 else []
                lidar_density = len(points) / 1000.0 if len(points) > 0 else 0.0
                
                # 3. Water depth (distance sensor or LiDAR z offsets)
                # Query distance sensor pointing downwards
                dist_data = self.client.getDistanceSensorData(distance_sensor_name="Distance1", vehicle_name="SimpleFlight")
                # If ground level is higher than expected or has specific reflectivity, estimate depth
                # Simplified: compare ground distance to standard 2m altitude
                water_depth = max(0.0, 2.0 - dist_data.distance) if dist_data else 0.0
                
            except Exception as e:
                print(f"[AirSimEnvAdapter] Sensor query failed, defaulting: {e}")
                rgb_smoke, lidar_density, water_depth = 0.0, 0.05, 0.0
                
        else:
            # Mock Mode Sensor Simulation
            # Base observations on user's mission configuration map choice
            print("[AirSimEnvAdapter] Simulating sensor readings for Mock mode...")
            expected_scenario = self.mission_config.get_map_scenario()
            
            if expected_scenario == "wildfire":
                rgb_smoke = 0.38 # Above threshold 0.15
                lidar_density = 0.05
                water_depth = 0.0
            elif expected_scenario == "flood":
                rgb_smoke = 0.0
                lidar_density = 0.02
                water_depth = 0.45 # Above threshold 0.2m
            else: # building_collapse
                rgb_smoke = 0.0
                lidar_density = 0.18 # Rubble density
                water_depth = 0.0

        # Run classification
        self.detected_scenario = self.detector.classify(lidar_density, rgb_smoke, water_depth)
        print(f"[AirSimEnvAdapter] Scenario successfully classified as: {self.detected_scenario}")
        
        # 2. Dynamically re-configure RL environment scenario settings
        config_path_dir = os.path.join(self.project_root, "uav_sar", "config")
        if self.is_mock:
            print(f"[AirSimEnvAdapter] Re-initializing Mock DisasterEnv to scenario '{self.detected_scenario}'")
            self.mock_env = DisasterEnv(
                env_config_path=os.path.join(config_path_dir, "env_config.yaml"),
                reward_config_path=os.path.join(config_path_dir, "reward_config.yaml"),
                scenario=self.detected_scenario,
                n_agents=4,
                difficulty="medium"
            )
            # Reset the mock env and sync maps
            self.mock_obs, self.mock_info = self.mock_env.reset()
            self.global_obstacle_map = self.mock_env.obstacle_map.copy()
            # If mock environment has custom victim manager settings, copy them
            self.mock_env.n_victims = len(self.victim_positions_3d)
            self.mock_env.victim_positions = [self.map_3d_to_2d(vx, vy) for vx, vy, vz in self.victim_positions_3d]
            self.mock_env.victim_found = np.zeros(len(self.victim_positions_3d), dtype=bool)
            
            # Recompute thermal map based on mock env features
            # Overlay floodwater/thermal features into wrapper local grids
            self.global_obstacle_map = self.mock_env.obstacle_map.copy()
        else:
            # Live AirSim environment settings
            pass
            
        self.drones_landed = True
        print("=== LANDING & SCENARIO DETECTION PHASE COMPLETE ===\n")

    def step(self):
        """
        Executes a single step. Inherits standard AirSimWrapper step operations,
        tracks drone history, coordinates GPS conversions via VictimTracker.
        """
        if not self.drones_landed:
            self.run_landing_and_detection()
            
        # Overwrite standard step updates
        self.step_count += 1
        positions_3d = self.get_drones_positions_3d()
        positions_2d = [self.map_3d_to_2d(x, y) for x, y, z in positions_3d]
        
        # Record trails for UI path plotting
        for i in range(4):
            key = f"drone_{i}"
            # Keep only the last 30 coordinates
            self.drone_trails[key].append(positions_3d[i][:2])
            if len(self.drone_trails[key]) > 50:
                self.drone_trails[key].pop(0)

        # 1. Update obstacles using LiDAR readings
        for i in range(4):
            name = "SimpleFlight" if i==0 else f"drone_{i}"
            if self.is_mock:
                self.global_obstacle_map = self.mock_env.obstacle_map.copy()
            else:
                try:
                    lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=name)
                    self.global_obstacle_map = self.lidar_processor.process_point_cloud(
                        lidar_data.point_cloud, positions_3d[i], self.global_obstacle_map
                    )
                except Exception as e:
                    pass

        # 2. Update thermal sensor readings
        thermal_grid = self.thermal_sensor.get_thermal_readings(positions_3d[0], self.victim_positions_3d)
        for i in range(1, 4):
            thermal_grid = np.maximum(thermal_grid, self.thermal_sensor.get_thermal_readings(positions_3d[i], self.victim_positions_3d))

        # 3. Mark victims as found & log GPS coordinates using VictimTracker
        for idx, (vx, vy, vz) in enumerate(self.victim_positions_3d):
            vr, vc = self.map_3d_to_2d(vx, vy)
            for ar, ac in positions_2d:
                dist = abs(ar - vr) + abs(ac - vc)
                if dist <= self.thermal_radius:
                    if not self.victim_found[idx]:
                        self.victim_found[idx] = True
                        
                        # Use VictimTracker to perform the GPS math, console print, and file logging
                        victim_details = self.tracker.record_victim_found(idx, vx, vy, self.step_count)
                        
                        # In-game HUD popups on Unreal Engine
                        if HAS_AIRSIM and not self.is_mock:
                            try:
                                lat = victim_details["latitude"]
                                lon = victim_details["longitude"]
                                dist_to_base = victim_details["distance_to_base"]
                                self.client.simPrintLogMessage(
                                    f"[HUD] TARGET LOCATED!", 
                                    f"Victim {idx} found! GPS: Lat {lat:.6f}, Lon {lon:.6f} | Dist: {dist_to_base:.1f}m", 
                                    3
                                )
                            except:
                                pass
                                
                        # Keep mock environment synced
                        if self.is_mock:
                            self.mock_env.victim_found[idx] = True

        # 4. Update coverage map
        for r, c in positions_2d:
            for dr in range(-self.obs_radius, self.obs_radius + 1):
                for dc in range(-self.obs_radius, self.obs_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.global_coverage_map[nr, nc] = 1.0

        # 5. Select actions using VDN network weights + Tabu Search
        actions = {}
        for i in range(4):
            key = f"drone_{i}"
            obs = self.build_agent_obs(i, positions_2d, thermal_grid)
            mask = self.get_action_mask(i, positions_2d[i][0], positions_2d[i][1])
            
            obs_tensor = torch_tensor = torch_tensor = None
            
            # Dynamic torch load inside function to ensure portability
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_vals = self.networks[i](obs_tensor, mask_tensor).squeeze(0)
                
            sorted_actions = torch.argsort(q_vals, descending=True).cpu().numpy()
            best_action = None
            r, c = positions_2d[i]
            
            for action in sorted_actions:
                if q_vals[action].item() < -1e8:
                    continue
                dr, dc = DisasterEnv.ACTION_DELTAS[action]
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.pos_history[key]:
                    continue
                else:
                    best_action = action
                    break
                    
            if best_action is None:
                for action in sorted_actions:
                    if q_vals[action].item() > -1e8:
                        best_action = action
                        break
                        
            dr, dc = DisasterEnv.ACTION_DELTAS[best_action]
            self.pos_history[key].append((r + dr, c + dc))
            if len(self.pos_history[key]) > 4:
                self.pos_history[key].pop(0)
                
            actions[key] = best_action

        # 6. Execute physical/mock movements
        deltas = DisasterEnv.ACTION_DELTAS
        for i in range(4):
            key = f"drone_{i}"
            name = "SimpleFlight" if i==0 else f"drone_{i}"
            action = actions[key]
            dr, dc = deltas[action]
            
            vx_2d = dr * self.cell_size_meters
            vy_2d = dc * self.cell_size_meters
            
            vx_2d, vy_2d = self.proximity_override(i, vx_2d, vy_2d, positions_3d)

            current_pos = positions_3d[i]
            target_pos = [current_pos[0] + vx_2d, current_pos[1] + vy_2d, self.altitude]
            dynamic_obstacles = [positions_3d[j] for j in range(4) if j != i]
            
            safe_vel = self.apf_3d.get_safe_velocity(current_pos, target_pos, dynamic_obstacles, max_speed=5.0)
            
            vx = safe_vel[0]
            vy = safe_vel[1]
            vz = safe_vel[2]

            if self.is_mock:
                mock_actions = {f"agent_{k}": actions[f"drone_{k}"] for k in range(4)}
                self.mock_obs, rewards, term, trunc, self.mock_info = self.mock_env.step(mock_actions)
                self.total_collisions = self.mock_info["total_collisions"]
            else:
                self.client.moveByVelocityAsync(vx, vy, vz, duration=1.0, vehicle_name=name)

        self.agent_battery -= 1.0

        navigable_cells = max(1, (self.global_obstacle_map == 0).sum())
        explored_cells = (self.global_coverage_map[self.global_obstacle_map == 0] > 0).sum()
        coverage_pct = (explored_cells / navigable_cells) * 100.0

        # Gather state object to broadcast via Websocket
        state_update = {
            "step": self.step_count,
            "coverage_pct": float(coverage_pct),
            "victims_found_count": int(self.victim_found.sum()),
            "total_victims_count": len(self.victim_positions_3d),
            "total_collisions": int(self.total_collisions),
            "detected_scenario": self.detected_scenario,
            "is_mock": self.is_mock,
            "drones": [
                {
                    "id": i,
                    "name": "SimpleFlight" if i==0 else f"drone_{i}",
                    "pos_3d": positions_3d[i],
                    "pos_2d": [int(positions_2d[i][0]), int(positions_2d[i][1])],
                    "battery": float(self.agent_battery[i]),
                    "trail": self.drone_trails[f"drone_{i}"]
                } for i in range(4)
            ],
            "found_victims": self.tracker.get_found_victims_list(),
            "obstacle_grid": self.global_obstacle_map.tolist(),
            "coverage_grid": self.global_coverage_map.tolist()
        }
        
        return state_update
