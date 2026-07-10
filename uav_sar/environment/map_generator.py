import numpy as np
from typing import Dict
from .scenarios.building_collapse import BuildingCollapseScenario
from .scenarios.wildfire import WildfireScenario
from .scenarios.flood import FloodScenario


class MapGenerator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.grid_size = cfg["grid_size"]
        self.scenarios = {
            "building_collapse": BuildingCollapseScenario(cfg),
            "wildfire": WildfireScenario(cfg),
            "flood": FloodScenario(cfg),
        }

    def generate(self, scenario: str, diff_params: Dict) -> np.ndarray:
        gen = self.scenarios[scenario]
        raw_grid = gen.generate(diff_params["obstacle_density"])
        obstacle_map = raw_grid.astype(np.float32)

        # Convert flooded cells (value 2) to 0.5 (restricted/slow-movement zone)
        obstacle_map[obstacle_map == 2.0] = 0.5

        # Always clear corners for agent spawning
        for corner in [(0, 0), (0, self.grid_size - 1),
                       (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]:
            r, c = corner
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        obstacle_map[nr, nc] = 0.0
                        
        # Extract thermal map from scenario if available (e.g. wildfire)
        if hasattr(gen, 'fire_map'):
            fm = gen.fire_map
            self.env_thermal_map = (fm / fm.max()) if fm.max() > 0 else fm
        else:
            self.env_thermal_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Extract smoke map from scenario if available (e.g. wildfire)
        if hasattr(gen, 'smoke_map'):
            self.env_smoke_map = gen.smoke_map.copy()
        else:
            self.env_smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
        return obstacle_map
