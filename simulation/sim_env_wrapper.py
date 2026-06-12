import os
import sys
import numpy as np
import yaml

# Add parent directory and uav_sar directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uav_sar")))

from uav_sar.environment.disaster_env import DisasterEnv
from uav_sar.agents.uav_agent import UAVAgent

class SimEnvWrapper:
    """
    Pure Python environment wrapper for the browser 2D simulation.
    Bypasses AirSim entirely. Uses the frozen 2D model and environment.
    """
    def __init__(self, map_environment: str, victim_positions: list):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # Determine RL scenario from map_environment
        mapping = {
            "SAR_Collapse": "building_collapse",
            "SAR_Wildfire": "wildfire",
            "SAR_Flood": "flood",
            "Building Collapse": "building_collapse",
            "Wildfire Zone": "wildfire",
            "Urban Flood": "flood",
            "building_collapse": "building_collapse",
            "wildfire": "wildfire",
            "flood": "flood"
        }
        self.scenario = mapping.get(map_environment, "building_collapse")
        self.victim_positions_2d = victim_positions  # List of [r, c]

        env_config_path = os.path.join(self.project_root, "uav_sar", "config", "env_config.yaml")
        reward_config_path = os.path.join(self.project_root, "uav_sar", "config", "reward_config.yaml")
        agent_config_path = os.path.join(self.project_root, "uav_sar", "config", "agent_config.yaml")
        checkpoint_path = os.path.join(self.project_root, "uav_sar", "checkpoints", "latest.pt")
        
        print("[SimEnvWrapper] Initializing DisasterEnv...", flush=True)
        # 1. Initialize the frozen DisasterEnv
        self.env = DisasterEnv(
            env_config_path=env_config_path,
            reward_config_path=reward_config_path,
            scenario=self.scenario,
            n_agents=4,
            difficulty="medium"
        )

        with open(agent_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        print("[SimEnvWrapper] Loading Agents...", flush=True)
        # 2. Load the 4 frozen UAV Agents
        self.agents = {}
        for i in range(4):
            print(f"[SimEnvWrapper] Loading agent {i}...", flush=True)
            agent = UAVAgent(
                obs_dim=166,
                n_actions=9,
                config=agent_cfg,
                device="cpu"  # Force CPU for stable web simulation
            )
            if os.path.exists(checkpoint_path):
                import torch
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                if f"agent_{i}" in ckpt:
                    agent.online_net.load_state_dict(ckpt[f"agent_{i}"]["online"])
                    agent.target_net.load_state_dict(ckpt[f"agent_{i}"]["target"])
                else:
                    agent.load(checkpoint_path)
            agent.online_net.eval()
            agent.target_net.eval()
            
            # Dynamically increase Tabu Search memory to force map coverage
            # (Done here to ensure the core 2D model architecture files remain completely untouched)
            agent.history_len = 50
            
            self.agents[f"agent_{i}"] = agent

        print("[SimEnvWrapper] Overriding victims...", flush=True)
        # Override victims to match user placement
        self.env.n_victims = len(self.victim_positions_2d)
        
        print("[SimEnvWrapper] Resetting env...", flush=True)
        # Reset to get initial state
        self.obs, self.info = self.env.reset()
        
        self.grid_size = self.env.grid_size
        self.realistic_coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        print("[SimEnvWrapper] Building thermal map...", flush=True)
        # Manually overwrite the random victims with our user-placed victims
        self.env.victim_positions = [tuple(v) for v in self.victim_positions_2d]
        self.env.victim_found = np.zeros(len(self.victim_positions_2d), dtype=bool)
        
        # Manually build the thermal map for the newly placed victims
        self.env.thermal_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for vr, vc in self.env.victim_positions:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    tr, tc = vr + dr, vc + dc
                    if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
                        dist = abs(dr) + abs(dc)
                        intensity = max(0.0, 1.0 - dist * 0.2)
                        self.env.thermal_map[tr, tc] = max(self.env.thermal_map[tr, tc], intensity)
        
        print("[SimEnvWrapper] Building initial obs...", flush=True)
        # Re-build initial observations to account for user placed victims
        self.obs = self.env._build_observations()

        self.step_count = 0
        
        # Pre-extract action mask indices based on obs dim definition
        # The mask is the last N_ACTIONS (9) elements
        self.mask_start = 166 - 9
        print("[SimEnvWrapper] Init complete!", flush=True)

    def step(self):
        """
        Executes one tick of the 2D simulation loop.
        Calls the frozen model: agent.select_action(state).
        """
        self.step_count += 1
        
        actions = {}
        for i in range(4):
            key = f"agent_{i}"
            state = self.obs[key]
            
            # Validate shape exactly as demanded by prompt
            assert state.shape == (166,), f"State shape mismatch: {state.shape}"
            
            action_mask = state[self.mask_start:]
            
            action = self.agents[key].select_action(state, action_mask, explore=False)
            actions[key] = action
            
        # Step the environment forward
        next_obs, rewards, term, trunc, info = self.env.step(actions)
        
        # --- 3D WRAPPER LAYER HOTFIX ---
        # The core 2D environment aggressively wipes a huge 11x11 square when a victim is found,
        # which erases neighboring victims in a cluster. We rebuild the thermal map for unfound
        # victims here in the wrapper so the 2D architecture remains strictly untouched.
        for rem_idx, (r_vr, r_vc) in enumerate(self.env.victim_positions):
            if not self.env.victim_found[rem_idx]:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        ntr, ntc = r_vr + dr, r_vc + dc
                        if 0 <= ntr < self.env.grid_size and 0 <= ntc < self.env.grid_size:
                            dist_v = abs(dr) + abs(dc)
                            intensity = max(0.0, 1.0 - dist_v * 0.2)
                            self.env.thermal_map[ntr, ntc] = max(self.env.thermal_map[ntr, ntc], intensity)
                            
        # Re-build the observations so the agents see the restored heat in the next tick!
        self.obs = self.env._build_observations()
        # -------------------------------
        
        # Update realistic coverage tracker purely for the 3D simulation metrics
        for i in range(4):
            pos_r, pos_c = self.env.agent_positions[i]
            self.realistic_coverage[pos_r, pos_c] = 1.0
            
        return {
            "step_count": self.step_count,
            "agent_positions": self.env.agent_positions.copy(),
            "victim_found": self.env.victim_found.copy(),
            "coverage_map": self.realistic_coverage.copy(),
            "obstacle_map": self.env.obstacle_map.copy(),
            "term": term,
            "trunc": trunc
        }
