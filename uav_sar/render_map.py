import numpy as np
from PIL import Image, ImageDraw
from environment.disaster_env import DisasterEnv

def render():
    env = DisasterEnv(
        env_config_path="config/env_config.yaml", 
        reward_config_path="config/reward_config.yaml", 
        difficulty="hard", 
        scenario="building_collapse"
    )
    env.reset()
    
    obstacle_map = env.obstacle_map
    h, w = obstacle_map.shape
    cell_size = 20
    
    img = Image.new('RGB', (w * cell_size, h * cell_size), color='white')
    draw = ImageDraw.Draw(img)
    
    for r in range(h):
        for c in range(w):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            if obstacle_map[r, c] == 1:
                # Obstacle
                draw.rectangle([x0, y0, x1, y1], fill='black')
            
            # grid lines
            draw.rectangle([x0, y0, x1, y1], outline='lightgray')
            
    # Draw victims
    for (vr, vc) in env.victim_positions:
        x0 = vc * cell_size
        y0 = vr * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        draw.rectangle([x0, y0, x1, y1], fill='red')

    # Draw agents starting positions
    for i, (ar, ac) in enumerate(env.agent_positions):
        x0 = ac * cell_size
        y0 = ar * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        draw.rectangle([x0, y0, x1, y1], fill='blue')

    img.save("map_render.png")
    print("Saved map_render.png")

if __name__ == "__main__":
    render()
