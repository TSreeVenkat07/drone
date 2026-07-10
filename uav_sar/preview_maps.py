import numpy as np
import matplotlib.pyplot as plt
from environment import DisasterEnv
import os

def save_map_preview(scenario_name: str, filename: str):
    # Initialize the environment with rgb_array render mode
    env = DisasterEnv(
        env_config_path="config/env_config.yaml",
        reward_config_path="config/reward_config.yaml",
        scenario=scenario_name,
        n_agents=4,
        difficulty="medium",
        render_mode="rgb_array"
    )
    env.reset()
    
    # Render the initial map frame
    img_array = env.render()
    
    # Save the image
    plt.imsave(filename, img_array)
    print(f"Saved {scenario_name} map preview to {filename}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    save_map_preview("building_collapse", "results/map_building_collapse.png")
    save_map_preview("wildfire", "results/map_wildfire.png")
    save_map_preview("flood", "results/map_flood.png")
    print("\nAll map previews generated! Check the 'results' folder.")
