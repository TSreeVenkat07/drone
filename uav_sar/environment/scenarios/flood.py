import numpy as np
from typing import Dict


class FloodScenario:
    """
    Flood Scenario: Urban flood disaster environment.
    Includes building blocks, roads, and open spaces.
    Some roads are flooded (restricted/slow-movement zones).
    Debris obstacles block certain road paths.
    Safe elevated dry land is on top of dry buildings.
    """
    def __init__(self, cfg: Dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()
        self.flood_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

    def generate(self, obstacle_density: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.flood_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # 1. Create a structured urban layout: roads and buildings
        road_width = 2
        block_size = 6
        cycle = road_width + block_size # 8
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                is_road_h = (r % cycle) < road_width
                is_road_v = (c % cycle) < road_width
                if not (is_road_h or is_road_v):
                    # Building block area
                    grid[r, c] = 1 # Mark as building/obstacle by default

        # 2. Simulate floodwater covering low-lying roads
        # We simulate a water source (e.g. river overflow) rising from one side or center
        center_r = self.rng.integers(10, 22)
        center_c = self.rng.integers(10, 22)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Distance-based water depth/flooding
                dist = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
                depth = np.clip(1.0 - dist / (self.grid_size * 0.4), 0.0, 1.0)
                self.flood_map[r, c] = depth

        # Mark flooded road cells as soft obstacles (represented as value 2 on the raw grid map)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] == 0 and self.flood_map[r, c] > 0.3:
                    # Flooded road
                    grid[r, c] = 2

        # 3. Add debris obstacles (value 1) randomly blocking roads
        n_debris = self.rng.integers(15, 30)
        for _ in range(n_debris):
            # Find a road cell (either flooded or dry)
            for _ in range(50):
                r = self.rng.integers(road_width, self.grid_size - road_width)
                c = self.rng.integers(road_width, self.grid_size - road_width)
                if grid[r, c] == 0 or grid[r, c] == 2:
                    grid[r, c] = 1 # Completely blocked by debris obstacle
                    break

        # Ensure boundaries are impassable walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
