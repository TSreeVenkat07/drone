import numpy as np
from typing import Dict


class FloodScenario:
    """
    Flood scenario: water-covered areas (impassable), islands of dry land,
    elevated ground near buildings. Victims stranded on isolated patches.
    Paths between islands are narrow and unstable.
    """
    def __init__(self, cfg: Dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()

    def generate(self, obstacle_density: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Water level simulation via Perlin-like noise
        water_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        # Multi-scale noise for realistic flood terrain
        for scale in [4, 8, 16]:
            freq = 1.0 / scale
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    water_map[r, c] += np.sin(r * freq * np.pi) * np.cos(c * freq * np.pi)

        # Normalize
        water_map = (water_map - water_map.min()) / (water_map.max() - water_map.min() + 1e-8)
        # Water covers low-lying areas
        grid[water_map < obstacle_density] = 1

        # Elevated island clusters (guaranteed dry land)
        n_islands = self.rng.integers(3, 7)
        for _ in range(n_islands):
            ir = self.rng.integers(4, self.grid_size - 4)
            ic = self.rng.integers(4, self.grid_size - 4)
            radius = self.rng.integers(2, 5)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr ** 2 + dc ** 2 <= radius ** 2:
                        nr, nc = ir + dr, ic + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            grid[nr, nc] = 0

        # Narrow bridges between islands (1-2 cell wide paths)
        n_bridges = self.rng.integers(2, 5)
        for _ in range(n_bridges):
            r1 = self.rng.integers(5, self.grid_size - 5)
            c1 = self.rng.integers(5, self.grid_size - 5)
            r2 = self.rng.integers(5, self.grid_size - 5)
            c2 = self.rng.integers(5, self.grid_size - 5)
            # Horizontal then vertical path
            for c in range(min(c1, c2), max(c1, c2) + 1):
                grid[r1, c] = 0
            for r in range(min(r1, r2), max(r1, r2) + 1):
                grid[r, c2] = 0

        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
