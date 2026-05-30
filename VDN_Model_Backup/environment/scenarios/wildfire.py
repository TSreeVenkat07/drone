import numpy as np
from typing import Dict


class WildfireScenario:
    """
    Wildfire scenario: open areas with fire spread patterns (impassable),
    smoke zones (reduced visibility), firebreaks (clear corridors).
    Victims scattered on elevated or isolated patches.
    """
    def __init__(self, cfg: Dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()

    def generate(self, obstacle_density: float) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Simulate fire spread from multiple ignition points
        n_fires = self.rng.integers(2, 5)
        fire_map = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for _ in range(n_fires):
            fr = self.rng.integers(5, self.grid_size - 5)
            fc = self.rng.integers(5, self.grid_size - 5)
            # Gaussian fire spread
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = ((r - fr) ** 2 + (c - fc) ** 2) ** 0.5
                    fire_map[r, c] += np.exp(-dist / (self.grid_size * 0.1))

        self.fire_map = fire_map
        
        # Threshold into obstacles
        threshold = np.percentile(fire_map, (1 - obstacle_density) * 100)
        grid[fire_map > threshold] = 1

        # Natural firebreaks (rivers / roads as clear corridors)
        n_breaks = self.rng.integers(1, 4)
        for _ in range(n_breaks):
            if self.rng.random() < 0.5:
                row = self.rng.integers(5, self.grid_size - 5)
                grid[row, :] = 0
            else:
                col = self.rng.integers(5, self.grid_size - 5)
                grid[:, col] = 0

        # Dense trees (impassable clusters) scattered around
        n_trees = int(obstacle_density * self.grid_size ** 2 * 0.3)
        for _ in range(n_trees):
            r = self.rng.integers(0, self.grid_size)
            c = self.rng.integers(0, self.grid_size)
            if self.rng.random() < 0.5:
                grid[r, c] = 1

        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
