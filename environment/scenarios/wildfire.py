import numpy as np
from typing import Dict


class WildfireScenario:
    """
    Wildfire Scenario: Urban fire disaster environment.
    Includes buildings, roads, and open spaces.
    Burning buildings act as fire hazard zones (no-fly obstacles).
    Smoke zones around fires cause sensor noise and movement drift.
    """
    def __init__(self, cfg: Dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()
        self.fire_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

    def generate(self, obstacle_density: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.fire_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        self.smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # 1. Create a structured urban layout: roads and buildings
        # Roads are placed at regular intervals, buildings fill the rest
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

        # 2. Introduce open spaces (parks / squares) by clearing some building blocks
        n_parks = self.rng.integers(2, 4)
        for _ in range(n_parks):
            pr = self.rng.integers(0, self.grid_size // cycle) * cycle + road_width
            pc = self.rng.integers(0, self.grid_size // cycle) * cycle + road_width
            grid[pr:pr+block_size, pc:pc+block_size] = 0

        # 3. Simulate unevenly distributed fire hazard zones on building structures
        # Pick 3-5 building coordinates as ignition sources
        n_fires = self.rng.integers(3, 6)
        fire_sources = []
        for _ in range(n_fires):
            # Try to find a building cell
            for _ in range(100):
                r = self.rng.integers(road_width, self.grid_size - road_width)
                c = self.rng.integers(road_width, self.grid_size - road_width)
                if grid[r, c] == 1:
                    fire_sources.append((r, c))
                    break
        
        # Compute thermal/fire spread from sources (Gaussian spread)
        for fr, fc in fire_sources:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = np.sqrt((r - fr) ** 2 + (c - fc) ** 2)
                    # Heat intensity
                    intensity = np.exp(-dist / (self.grid_size * 0.15))
                    self.fire_map[r, c] = max(self.fire_map[r, c], intensity)

        # Burning buildings (intensity > 0.6) are high-risk no-fly regions (obstacles)
        grid[self.fire_map > 0.6] = 1

        # 4. Generate smoke map around fire zones
        for fr, fc in fire_sources:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = np.sqrt((r - fr) ** 2 + (c - fc) ** 2)
                    smoke_intensity = np.exp(-dist / (self.grid_size * 0.25))
                    self.smoke_map[r, c] = max(self.smoke_map[r, c], smoke_intensity)

        # Ensure boundaries are impassable walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
