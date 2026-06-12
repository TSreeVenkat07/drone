"""
xBD (xView2) Image Processor and Grid Environment
Converts real xView2 post-disaster satellite imagery + JSON annotations
into grid-based environments for UAV SAR evaluation, with full GPS coordinate mapping.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    from shapely.geometry import shape, Polygon
except ImportError:
    shape = None
    Polygon = None

DAMAGE_LABELS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}


class GeoTransform:
    """Affine transform mapping pixel coordinates to GPS (lat, lon)."""

    def __init__(self, bounds: Dict, img_width: int, img_height: int):
        self.min_lon = bounds["min_lon"]
        self.max_lon = bounds["max_lon"]
        self.min_lat = bounds["min_lat"]
        self.max_lat = bounds["max_lat"]
        self.img_width = img_width
        self.img_height = img_height

    def pixel_to_gps(self, row: int, col: int) -> Tuple[float, float]:
        lon = self.min_lon + (col / self.img_width) * (self.max_lon - self.min_lon)
        lat = self.max_lat - (row / self.img_height) * (self.max_lat - self.min_lat)
        return (lat, lon)

    def grid_to_gps(self, grid_row: int, grid_col: int, grid_size: int) -> Tuple[float, float]:
        pixel_row = (grid_row + 0.5) * (self.img_height / grid_size)
        pixel_col = (grid_col + 0.5) * (self.img_width / grid_size)
        return self.pixel_to_gps(pixel_row, pixel_col)


class XBDImageProcessor:
    """Processes xBD post-disaster image + JSON annotations into a grid environment."""

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size

    def load_annotation(self, json_path: str) -> Dict:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def extract_bounds(self, annotation: Dict) -> Dict:
        meta = annotation.get("metadata", {})
        if "geo" in meta:
            geo = meta["geo"]
            if "bounds" in geo:
                b = geo["bounds"]
                return {"min_lon": b[0], "min_lat": b[1], "max_lon": b[2], "max_lat": b[3]}
            elif "lng" in geo and "lat" in geo:
                lng, lat = geo["lng"], geo["lat"]
                delta = 0.005
                return {"min_lon": lng - delta, "min_lat": lat - delta,
                        "max_lon": lng + delta, "max_lat": lat + delta}

        all_coords = []
        for feat in annotation.get("features", {}).get("xy", []):
            if "wkt" in feat.get("properties", {}):
                wkt = feat["properties"]["wkt"]
                coords = self._parse_wkt_coords(wkt)
                all_coords.extend(coords)

        if all_coords:
            lons = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            return {"min_lon": min(lons), "min_lat": min(lats),
                    "max_lon": max(lons), "max_lat": max(lats)}

        return {"min_lon": -90.0, "min_lat": 29.9, "max_lon": -89.99, "max_lat": 29.91}

    def _parse_wkt_coords(self, wkt_str: str) -> List[Tuple[float, float]]:
        coords = []
        try:
            inner = wkt_str.split("((")[1].split("))")[0]
            pairs = inner.split(",")
            for pair in pairs:
                parts = pair.strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
        except (IndexError, ValueError):
            pass
        return coords

    def extract_damage_polygons(self, annotation: Dict) -> List[Dict]:
        buildings = []
        features = annotation.get("features", {})
        xy_features = features.get("xy", [])
        if not xy_features and isinstance(features, list):
            xy_features = features

        for feat in xy_features:
            props = feat.get("properties", {})
            damage_str = props.get("subtype", props.get("damage", "un-classified"))
            damage_level = DAMAGE_LABELS.get(damage_str, 0)

            pixel_coords = []
            if "wkt" in props:
                pixel_coords = self._parse_wkt_coords(props["wkt"])
            elif "xy" in feat:
                xy = feat["xy"]
                if isinstance(xy, list) and len(xy) > 0:
                    if isinstance(xy[0], (list, tuple)):
                        pixel_coords = [(float(p[0]), float(p[1])) for p in xy]

            if pixel_coords and damage_level > 0:
                buildings.append({"damage_level": damage_level, "pixel_coords": pixel_coords})

        return buildings

    def rasterize_damage(self, buildings: List[Dict], img_width: int, img_height: int) -> np.ndarray:
        damage_map = np.zeros((img_height, img_width), dtype=np.float32)

        if Image is not None and ImageDraw is not None:
            for bldg in buildings:
                mask_img = Image.new("L", (img_width, img_height), 0)
                draw = ImageDraw.Draw(mask_img)
                coords = [(int(x), int(y)) for x, y in bldg["pixel_coords"]]
                if len(coords) >= 3:
                    draw.polygon(coords, fill=255)
                    mask = np.array(mask_img) > 0
                    damage_map[mask] = max(damage_map[mask].max(), bldg["damage_level"])
        else:
            for bldg in buildings:
                coords = bldg["pixel_coords"]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                x_min = max(0, min(x_min, img_width - 1))
                x_max = max(0, min(x_max, img_width - 1))
                y_min = max(0, min(y_min, img_height - 1))
                y_max = max(0, min(y_max, img_height - 1))
                damage_map[y_min:y_max+1, x_min:x_max+1] = bldg["damage_level"]

        return damage_map

    def downsample_damage(self, damage_map: np.ndarray) -> np.ndarray:
        h, w = damage_map.shape
        gh, gw = self.grid_size, self.grid_size
        cell_h, cell_w = h // gh, w // gw

        grid = np.zeros((gh, gw), dtype=np.float32)
        for r in range(gh):
            for c in range(gw):
                block = damage_map[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                if block.size > 0:
                    grid[r, c] = block.max()
        return grid

    def damage_to_env_grid(self, damage_grid: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        gs = self.grid_size
        obstacle_map = np.zeros((gs, gs), dtype=np.float32)
        victim_zones = []
        priority_zones = []

        for r in range(gs):
            for c in range(gs):
                dmg = damage_grid[r, c]
                if dmg == 4:
                    obstacle_map[r, c] = 0.0
                    victim_zones.append((r, c))
                elif dmg == 3:
                    obstacle_map[r, c] = 0.0
                    priority_zones.append((r, c))
                elif dmg >= 1:
                    obstacle_map[r, c] = 1.0
                else:
                    obstacle_map[r, c] = 0.0

        # Ensure corners are clear for agent spawning
        for corner_r, corner_c in [(0, 0), (0, gs-1), (gs-1, 0), (gs-1, gs-1)]:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = corner_r + dr, corner_c + dc
                    if 0 <= nr < gs and 0 <= nc < gs:
                        obstacle_map[nr, nc] = 0.0

        return obstacle_map, victim_zones, priority_zones

    def process_image(self, json_path: str, image_path: Optional[str] = None) -> Dict:
        annotation = self.load_annotation(json_path)
        bounds = self.extract_bounds(annotation)

        meta = annotation.get("metadata", {})
        img_meta = meta.get("img", {})
        img_width = img_meta.get("width", 1024)
        img_height = img_meta.get("height", 1024)

        if image_path and Image is not None and os.path.exists(image_path):
            with Image.open(image_path) as img:
                img_width, img_height = img.size

        geo_transform = GeoTransform(bounds, img_width, img_height)

        buildings = self.extract_damage_polygons(annotation)
        if buildings:
            damage_map = self.rasterize_damage(buildings, img_width, img_height)
            damage_grid = self.downsample_damage(damage_map)
        else:
            damage_grid = self._generate_synthetic_damage()

        obstacle_map, victim_zones, priority_zones = self.damage_to_env_grid(damage_grid)

        return {
            "obstacle_map": obstacle_map,
            "victim_zones": victim_zones,
            "priority_zones": priority_zones,
            "damage_grid": damage_grid,
            "geo_transform": geo_transform,
            "metadata": {
                "json_path": json_path, "image_path": image_path,
                "img_width": img_width, "img_height": img_height,
                "bounds": bounds,
                "n_buildings": len(buildings),
                "n_destroyed": sum(1 for b in buildings if b["damage_level"] == 4),
                "n_major": sum(1 for b in buildings if b["damage_level"] == 3),
            }
        }

    def _generate_synthetic_damage(self) -> np.ndarray:
        gs = self.grid_size
        damage = np.zeros((gs, gs), dtype=np.float32)
        rng = np.random.default_rng(42)
        for r in range(gs):
            for c in range(gs):
                if (r % 4 in [1, 2]) and (c % 4 in [1, 2]):
                    damage[r, c] = rng.choice([1, 2])
        center = gs // 2
        for r in range(center - 4, center + 4):
            for c in range(center - 4, center + 4):
                if 0 <= r < gs and 0 <= c < gs:
                    if rng.random() < 0.5:
                        damage[r, c] = 4
                    else:
                        damage[r, c] = 3
        return damage


# ─── Standalone Grid Environment for xBD Evaluation ───

class XBDGridEnvironment:
    """
    Multi-UAV grid environment built from real xBD damage data.
    Self-contained. Compatible observation format with existing DuelingDQN model.
    """

    ACTION_DELTAS = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
        ( 0,  0),
    ]
    N_ACTIONS = 9

    def __init__(self, processed_data: Dict, n_agents: int = 4,
                 n_victims: int = 7, max_steps: int = 500,
                 obs_radius: int = 5, thermal_radius: int = 2,
                 battery_max: int = 500):
        self.grid_size = processed_data["obstacle_map"].shape[0]
        self.base_obstacle_map = processed_data["obstacle_map"].copy()
        self.victim_zones = processed_data["victim_zones"]
        self.priority_zones = processed_data["priority_zones"]
        self.damage_grid = processed_data["damage_grid"]
        self.geo_transform = processed_data["geo_transform"]
        self.image_metadata = processed_data["metadata"]

        self.n_agents = n_agents
        self.n_victims = n_victims
        self.max_steps = max_steps
        self.obs_radius = obs_radius
        self.thermal_radius = thermal_radius
        self.battery_max = battery_max

        local_dim = (2 * obs_radius + 1) ** 2
        thermal_dim = (2 * thermal_radius + 1) ** 2
        pos_dim = 2
        other_agents_dim = (n_agents - 1) * 2
        extras_dim = 3
        mask_dim = self.N_ACTIONS
        self.obs_dim = local_dim + thermal_dim + pos_dim + other_agents_dim + extras_dim + mask_dim

        self.uav_path_log = []
        self.victim_gps_log = []
        self.victim_ground_truth = []

        self.obstacle_map = None
        self.coverage_map = None
        self.thermal_map = None
        self.smoke_map = None
        self.victim_positions = None
        self.victim_found = None
        self.agent_positions = None
        self.agent_battery = None
        self.step_count = 0
        self.collision_counts = None
        self.victim_detection_times = {}
        self.coverage_milestone_achieved = set()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(seed)

        self.step_count = 0
        self.coverage_milestone_achieved = set()
        self.victim_detection_times = {}
        self.uav_path_log = []
        self.victim_gps_log = []
        self.victim_ground_truth = []

        self.obstacle_map = self.base_obstacle_map.copy()
        gs = self.grid_size

        if self.victim_zones:
            n_to_place = min(self.n_victims, len(self.victim_zones))
            chosen_idx = rng.choice(len(self.victim_zones), n_to_place, replace=False)
            self.victim_positions = [self.victim_zones[i] for i in chosen_idx]
        else:
            passable = [(r, c) for r in range(gs) for c in range(gs) if self.obstacle_map[r, c] == 0]
            if len(passable) >= self.n_victims:
                chosen_idx = rng.choice(len(passable), self.n_victims, replace=False)
                self.victim_positions = [passable[i] for i in chosen_idx]
            else:
                self.victim_positions = passable[:self.n_victims]

        while len(self.victim_positions) < self.n_victims:
            passable = [(r, c) for r in range(gs) for c in range(gs)
                        if self.obstacle_map[r, c] == 0 and (r, c) not in self.victim_positions]
            if passable:
                idx = rng.integers(0, len(passable))
                self.victim_positions.append(passable[idx])
            else:
                break

        self.victim_found = np.zeros(len(self.victim_positions), dtype=bool)

        for v_idx, (vr, vc) in enumerate(self.victim_positions):
            lat, lon = self.geo_transform.grid_to_gps(vr, vc, gs)
            dmg_level = int(self.damage_grid[vr, vc])
            self.victim_ground_truth.append({
                "victim_id": v_idx, "grid_row": vr, "grid_col": vc,
                "lat": lat, "lon": lon, "damage_level": dmg_level,
            })

        self.thermal_map = np.zeros((gs, gs), dtype=np.float32)
        for vr, vc in self.victim_positions:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = vr + dr, vc + dc
                    if 0 <= nr < gs and 0 <= nc < gs:
                        dist = (dr ** 2 + dc ** 2) ** 0.5
                        intensity = np.exp(-dist / 2.0) * 0.7
                        noise = rng.normal(0, 0.035)
                        self.thermal_map[nr, nc] = min(1.0, self.thermal_map[nr, nc] + intensity + noise)

        for pr, pc in self.priority_zones:
            if 0 <= pr < gs and 0 <= pc < gs:
                self.thermal_map[pr, pc] = min(1.0, self.thermal_map[pr, pc] + 0.2)

        noise_cells = int(0.03 * gs * gs)
        for _ in range(noise_cells):
            r, c = rng.integers(0, gs), rng.integers(0, gs)
            self.thermal_map[r, c] = min(1.0, self.thermal_map[r, c] + rng.uniform(0.1, 0.3))

        self.smoke_map = np.zeros((gs, gs), dtype=np.float32)
        self.coverage_map = np.zeros((gs, gs), dtype=np.float32)

        self.agent_positions = self._spawn_agents()
        self.agent_battery = np.full(self.n_agents, self.battery_max, dtype=np.float32)
        self.collision_counts = np.zeros(self.n_agents, dtype=int)
        self._update_coverage()

        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            lat, lon = self.geo_transform.grid_to_gps(r, c, gs)
            self.uav_path_log.append((0, i, r, c, lat, lon))

        obs = self._build_observations()
        info = {"action_masks": self._get_all_action_masks()}
        return obs, info

    def step(self, actions: Dict[str, int]):
        self.step_count += 1
        rewards = {f"agent_{i}": 0.0 for i in range(self.n_agents)}
        gs = self.grid_size

        for i in range(self.n_agents):
            key = f"agent_{i}"
            action = actions[key]
            r, c = self.agent_positions[i]

            dr, dc = self.ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc

            if not (0 <= nr < gs and 0 <= nc < gs):
                self.collision_counts[i] += 1
                continue
            if self.obstacle_map[nr, nc] == 1:
                self.collision_counts[i] += 1
                continue

            self.agent_positions[i] = np.array([nr, nc])

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    self.collision_counts[i] += 1
                    self.collision_counts[j] += 1

        self._update_coverage()
        # Coverage: only count passable cells, cap at 100%
        passable_mask = self.obstacle_map < 1
        covered_passable = float((self.coverage_map[passable_mask] > 0).sum())
        passable_cells = float(passable_mask.sum())
        coverage_pct = min(100.0, (covered_passable / max(passable_cells, 1)) * 100)

        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            for v_idx, (vr, vc) in enumerate(self.victim_positions):
                dist = abs(r - vr) + abs(c - vc)
                thermal_val = self.thermal_map[r, c] if 0 <= r < gs and 0 <= c < gs else 0
                if dist <= self.thermal_radius and thermal_val > 0.1:
                    if not self.victim_found[v_idx]:
                        self.victim_found[v_idx] = True
                        self.victim_detection_times[v_idx] = self.step_count

                        lat, lon = self.geo_transform.grid_to_gps(vr, vc, gs)
                        dmg = int(self.damage_grid[vr, vc])
                        self.victim_gps_log.append({
                            "victim_id": v_idx, "grid_row": vr, "grid_col": vc,
                            "lat": lat, "lon": lon, "damage_level": dmg,
                            "found_by_agent": i, "found_at_step": self.step_count,
                        })

                        for dr2 in range(-5, 6):
                            for dc2 in range(-5, 6):
                                ntr, ntc = vr + dr2, vc + dc2
                                if 0 <= ntr < gs and 0 <= ntc < gs:
                                    self.thermal_map[ntr, ntc] = 0.0

        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            lat, lon = self.geo_transform.grid_to_gps(r, c, gs)
            self.uav_path_log.append((self.step_count, i, r, c, lat, lon))

        for i in range(self.n_agents):
            self.agent_battery[i] -= 1.0

        terminated = bool(self.victim_found.all())
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._build_observations()
        info = {
            "action_masks": self._get_all_action_masks(),
            "victims_found": int(self.victim_found.sum()),
            "total_collisions": int(self.collision_counts.sum()),
            "coverage_pct": float(coverage_pct),
            "step_count": self.step_count,
        }

        return obs, rewards, terminated, truncated, info

    def _spawn_agents(self) -> list:
        positions = []
        gs = self.grid_size
        corners = [(2, 2), (2, gs - 3), (gs - 3, 2), (gs - 3, gs - 3)]
        for i in range(self.n_agents):
            cr, cc = corners[i % len(corners)]
            placed = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = cr + dr, cc + dc
                    if (0 <= r < gs and 0 <= c < gs and self.obstacle_map[r, c] == 0):
                        positions.append(np.array([r, c]))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                positions.append(np.array([cr, cc]))
        return positions

    def _update_coverage(self) -> float:
        prev = float(self.coverage_map.sum())
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            for dr in range(-self.obs_radius, self.obs_radius + 1):
                for dc in range(-self.obs_radius, self.obs_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.coverage_map[nr, nc] = min(1.0, self.coverage_map[nr, nc] + 0.5)
        return float(self.coverage_map.sum()) - prev

    def _get_action_mask(self, agent_idx: int) -> np.ndarray:
        mask = np.ones(self.N_ACTIONS, dtype=bool)
        r, c = self.agent_positions[agent_idx]
        for a, (dr, dc) in enumerate(self.ACTION_DELTAS[:8]):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                mask[a] = False
            elif self.obstacle_map[nr, nc] == 1:
                mask[a] = False
        return mask

    def _get_all_action_masks(self) -> Dict[str, np.ndarray]:
        return {f"agent_{i}": self._get_action_mask(i) for i in range(self.n_agents)}

    def _build_single_obs(self, agent_idx: int) -> np.ndarray:
        r, c = self.agent_positions[agent_idx]
        gs = self.grid_size

        pad = self.obs_radius
        local = np.full((2 * pad + 1, 2 * pad + 1), -1.0, dtype=np.float32)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    local[dr + pad, dc + pad] = float(self.obstacle_map[nr, nc])

        tp = self.thermal_radius
        thermal = np.zeros((2 * tp + 1, 2 * tp + 1), dtype=np.float32)
        for dr in range(-tp, tp + 1):
            for dc in range(-tp, tp + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    thermal[dr + tp, dc + tp] = self.thermal_map[nr, nc]

        pos = np.array([r / gs, c / gs], dtype=np.float32)

        other = []
        for j in range(self.n_agents):
            if j != agent_idx:
                or_, oc = self.agent_positions[j]
                other.extend([(or_ - r) / gs, (oc - c) / gs])
        other = np.array(other, dtype=np.float32)

        battery_ratio = self.agent_battery[agent_idx] / self.battery_max
        step_ratio = self.step_count / self.max_steps
        victim_ratio = float(self.victim_found.sum()) / max(len(self.victim_positions), 1)
        extras = np.array([battery_ratio, step_ratio, victim_ratio], dtype=np.float32)

        mask = self._get_action_mask(agent_idx).astype(np.float32)

        return np.concatenate([local.flatten(), thermal.flatten(), pos, other, extras, mask])

    def _build_observations(self) -> Dict[str, np.ndarray]:
        return {f"agent_{i}": self._build_single_obs(i) for i in range(self.n_agents)}
