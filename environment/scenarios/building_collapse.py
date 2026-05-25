import numpy as np
from typing import Dict


class BuildingCollapseScenario:
    """
    Urban building collapse: dense rubble clusters, blocked corridors,
    collapsed walls forming maze-like narrow passages.
    """
    def __init__(self, cfg: Dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()

    def generate(self, obstacle_density: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Structural walls (building outlines)
        n_buildings = 4
        for _ in range(n_buildings):
            br = self.rng.integers(3, self.grid_size - 8)
            bc = self.rng.integers(3, self.grid_size - 8)
            bh = self.rng.integers(4, 9)
            bw = self.rng.integers(4, 9)
            # Draw building perimeter
            grid[br:br + bh, bc] = 1
            grid[br:br + bh, bc + bw] = 1
            grid[br, bc:bc + bw] = 1
            grid[br + bh, bc:bc + bw + 1] = 1
            # Add doorway (gap in walls)
            door_wall = self.rng.choice(["N", "S", "E", "W"])
            if door_wall == "N":
                dc = self.rng.integers(bc + 1, bc + bw)
                grid[br, dc] = 0
            elif door_wall == "S":
                dc = self.rng.integers(bc + 1, bc + bw)
                grid[br + bh, dc] = 0
            elif door_wall == "E":
                dr = self.rng.integers(br + 1, br + bh)
                grid[dr, bc + bw] = 0
            else:
                dr = self.rng.integers(br + 1, br + bh)
                grid[dr, bc] = 0

        # Collapsed rubble (dense irregular clusters)
        n_rubble = int(obstacle_density * self.grid_size ** 2 * 0.6)
        for _ in range(n_rubble):
            r = self.rng.integers(0, self.grid_size)
            c = self.rng.integers(0, self.grid_size)
            size = self.rng.integers(1, 4)
            for dr in range(size):
                for dc in range(size):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.rng.random() < obstacle_density + 0.1:
                            grid[nr, nc] = 1

        # Ensure border walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
