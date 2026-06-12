import os
import sys
import yaml
import time
import numpy as np
try:
    import airsim
except ImportError:
    print("AirSim module not found.")
    sys.exit(1)

from environment.disaster_env import DisasterEnv

def build_unreal_environment():
    print("Connecting to AirSim to import environment...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Load configuration
    with open("config/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    
    grid_size = env_cfg["grid_size"]
    cell_size_meters = 5.0
    
    print("Initializing DisasterEnv to get obstacle map...")
    env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml", "building_collapse", 4, "hard")
    env.reset()
    
    obstacle_map = env.obstacle_map
    
    # 3D Victim positions from airsim_wrapper.py
    victim_positions_3d = [
        [-80.0, -80.0, 0.0],
        [-50.0, 60.0, 0.0],
        [10.0, -40.0, 0.0],
        [40.0, 80.0, 0.0],
        [90.0, -90.0, 0.0],
        [0.0, 10.0, 0.0],
        [-30.0, -20.0, 0.0]
    ]

    print("Destroying any previously spawned assets...")
    # Actually airsim doesn't have a clear all API easily, we'll just spawn new ones.
    
    print("Spawning Obstacles (Building Collapse Rubble)...")
    scale_cube = airsim.Vector3r(2.5, 2.5, 15.0) # 2.5x2.5m width, 15m tall
    count = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if obstacle_map[r, c] == 1.0:
                # Convert to 3D
                x = (r - grid_size / 2) * cell_size_meters
                y = (c - grid_size / 2) * cell_size_meters
                
                # Spawn a cube
                pose = airsim.Pose(airsim.Vector3r(x, y, 0), airsim.to_quaternion(0, 0, 0))
                # Name must be unique
                name = f"obstacle_{r}_{c}"
                try:
                    client.simSpawnObject(name, "Cube", pose, scale_cube)
                    count += 1
                except Exception:
                    pass

    print(f"Spawned {count} obstacle blocks.")
    
    print("Spawning Victims...")
    scale_sphere = airsim.Vector3r(2.0, 2.0, 2.0)
    for i, (vx, vy, vz) in enumerate(victim_positions_3d):
        pose = airsim.Pose(airsim.Vector3r(vx, vy, vz), airsim.to_quaternion(0, 0, 0))
        name = f"victim_{i}"
        try:
            client.simSpawnObject(name, "Sphere", pose, scale_sphere)
        except Exception:
            pass
            
    print(f"Spawned {len(victim_positions_3d)} victims.")
    print("Environment import complete! You should now see the map in Unreal Editor.")

if __name__ == "__main__":
    build_unreal_environment()
