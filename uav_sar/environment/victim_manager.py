import numpy as np
from typing import Tuple, List


class VictimManager:
    def __init__(self, cfg: dict):
        self.grid_size = cfg["grid_size"]
        self.rng = np.random.default_rng()

    def place_victims(
        self, obstacle_map: np.ndarray, n_victims: int, visibility: float
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Place victims in passable cells and build thermal emission map."""
        passable = np.argwhere(obstacle_map == 0)
        if len(passable) < n_victims:
            raise ValueError("Not enough passable cells for victims.")
        chosen = self.rng.choice(len(passable), n_victims, replace=False)
        victim_positions = [(int(passable[i][0]), int(passable[i][1])) for i in chosen]

        # Build thermal map with FEMA-based emission probability model
        thermal = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for vr, vc in victim_positions:
            # Primary heat signature (strong near victim)
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = vr + dr, vc + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        dist = (dr ** 2 + dc ** 2) ** 0.5
                        intensity = np.exp(-dist / 2.0) * visibility
                        # Add noise for realism
                        noise = self.rng.normal(0, 0.05 * (1 - visibility))
                        thermal[nr, nc] = min(1.0, thermal[nr, nc] + intensity + noise)

        # Background thermal noise (false positive sources)
        noise_cells = int(0.03 * self.grid_size ** 2)
        for _ in range(noise_cells):
            r = self.rng.integers(0, self.grid_size)
            c = self.rng.integers(0, self.grid_size)
            thermal[r, c] = min(1.0, thermal[r, c] + self.rng.uniform(0.1, 0.3))

        return victim_positions, thermal
