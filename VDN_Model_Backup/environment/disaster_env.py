import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import yaml
import os
from .map_generator import MapGenerator
from .victim_manager import VictimManager


class DisasterEnv(gym.Env):
    """
    Multi-UAV Search and Rescue Environment.
    3 scenarios: building_collapse, wildfire, flood.
    Supports action masking to eliminate invalid moves.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    ACTION_DELTAS = [
        (-1,  0),  # N
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
        ( 0,  0),  # Hover
    ]
    N_ACTIONS = 9

    def __init__(
        self,
        env_config_path: str = "config/env_config.yaml",
        reward_config_path: str = "config/reward_config.yaml",
        scenario: str = "building_collapse",
        n_agents: int = 4,
        difficulty: str = "medium",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        with open(env_config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(reward_config_path) as f:
            self.rcfg = yaml.safe_load(f)

        self.scenario = scenario
        self.n_agents = n_agents
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.grid_size = self.cfg["grid_size"]
        self.n_victims = self.cfg["n_victims"]
        self.max_steps = self.cfg["max_steps"]
        self.obs_radius = self.cfg["local_obs_radius"]
        self.thermal_radius = self.cfg["thermal_radius"]

        # Dimensions for single agent observation
        local_dim = (2 * self.obs_radius + 1) ** 2       # occupancy 11x11=121
        thermal_dim = (2 * self.thermal_radius + 1) ** 2 # thermal 5x5=25
        pos_dim = 2
        other_agents_dim = (self.n_agents - 1) * 2
        extras_dim = 3   # battery, step_ratio, victim_found_ratio
        mask_dim = self.N_ACTIONS

        self.obs_dim = local_dim + thermal_dim + pos_dim + other_agents_dim + extras_dim + mask_dim

        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
            for i in range(self.n_agents)
        })
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(self.N_ACTIONS)
            for i in range(self.n_agents)
        })

        self.map_gen = MapGenerator(self.cfg)
        self.victim_mgr = VictimManager(self.cfg)

        # State
        self.obstacle_map = None
        self.agent_positions = None
        self.coverage_map = None
        self.thermal_map = None
        self.victim_positions = None
        self.victim_found = None
        self.agent_battery = None
        self.step_count = 0
        self.collision_counts = None
        self.victim_detection_times = {}
        self.coverage_milestone_achieved = set()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.coverage_milestone_achieved = set()
        self.victim_detection_times = {}

        diff_params = self.cfg["difficulties"][self.difficulty]
        self.obstacle_map = self.map_gen.generate(self.scenario, diff_params)
        self.victim_positions, self.thermal_map = self.victim_mgr.place_victims(
            self.obstacle_map, self.n_victims, diff_params["victim_visibility"]
        )
        
        # Overlay environmental thermal sources (e.g. wildfire)
        if hasattr(self.map_gen, 'env_thermal_map'):
            self.thermal_map = np.clip(self.thermal_map + self.map_gen.env_thermal_map, 0.0, 1.0)
            
        self.victim_found = np.zeros(self.n_victims, dtype=bool)
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_positions = self._spawn_agents()
        self.agent_battery = np.full(self.n_agents, self.cfg["agent_battery_max"], dtype=np.float32)
        self.collision_counts = np.zeros(self.n_agents, dtype=int)
        self._update_coverage()

        obs = self._build_observations()
        info = {"action_masks": self._get_all_action_masks()}
        return obs, info

    # ------------------------------------------------------------------ step
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        self.step_count += 1
        rewards = {f"agent_{i}": 0.0 for i in range(self.n_agents)}
        prev_positions = {i: tuple(self.agent_positions[i]) for i in range(self.n_agents)}
        prev_coverage = float(self.coverage_map.sum())

        # Move agents, compute per-agent reward contributions
        for i in range(self.n_agents):
            key = f"agent_{i}"
            action = actions[key]
            dr, dc = self.ACTION_DELTAS[action]
            r, c = self.agent_positions[i]
            nr, nc = r + dr, c + dc

            # --- TIER 2: Safety (collision penalties, severe) ---
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                # rewards[key] += self.rcfg["tier2_safety"]["out_of_bounds"]
                self.collision_counts[i] += 1
                continue

            if self.obstacle_map[nr, nc] == 1:
                # Wall collision - severity escalates after repeated collisions
                base = self.rcfg["tier2_safety"]["wall_collision"]
                if self.collision_counts[i] >= self.rcfg["tier2_safety"]["repeated_collision_threshold"]:
                    base *= self.rcfg["tier2_safety"]["repeated_collision_multiplier"]
                # rewards[key] += base
                self.collision_counts[i] += 1
                continue

            # Soft near-wall penalty
            for dr2, dc2 in self.ACTION_DELTAS[:8]:
                nnr, nnc = nr + dr2, nc + dc2
                if 0 <= nnr < self.grid_size and 0 <= nnc < self.grid_size:
                    if self.obstacle_map[nnr, nnc] == 1:
                        # rewards[key] += self.rcfg["tier2_safety"]["near_wall_soft"]
                        break

            # Agent-agent collision check (positions after moves are applied)
            self.agent_positions[i] = np.array([nr, nc])

        # Agent-agent collision detection (post-move)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    # rewards[f"agent_{i}"] += self.rcfg["tier2_safety"]["agent_collision"]
                    # rewards[f"agent_{j}"] += self.rcfg["tier2_safety"]["agent_collision"]
                    self.collision_counts[i] += 1
                    self.collision_counts[j] += 1
                else:
                    dist = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                    if dist < 3.0:
                        # rewards[f"agent_{i}"] += self.rcfg["tier2_safety"]["near_agent_soft"]
                        # rewards[f"agent_{j}"] += self.rcfg["tier2_safety"]["near_agent_soft"]
                        pass

        # --- TIER 3: Coverage rewards ---
        new_coverage = self._update_coverage()
        total_coverage = float(self.coverage_map.sum())
        new_cells = total_coverage - prev_coverage
        shared_coverage_reward = new_cells * self.rcfg["tier3_coverage"]["new_cell"] / max(self.n_agents, 1)
        frontier_cells = self._count_frontier_cells()
        for i in range(self.n_agents):
            key = f"agent_{i}"
            # rewards[key] += shared_coverage_reward
            # Frontier bonus proportional to agent's frontier coverage
            if frontier_cells > 0:
                # rewards[key] += self.rcfg["tier3_coverage"]["frontier_bonus"] * 0.1
                pass

        # Coverage milestones (shared global reward)
        coverage_pct = total_coverage / (self.grid_size ** 2) * 100
        for milestone, rwd_key in [(25, "milestone_25pct"), (50, "milestone_50pct"),
                                    (75, "milestone_75pct"), (100, "milestone_100pct")]:
            if coverage_pct >= milestone and milestone not in self.coverage_milestone_achieved:
                self.coverage_milestone_achieved.add(milestone)
                split = self.rcfg["tier3_coverage"][rwd_key] / self.n_agents
                for i in range(self.n_agents):
                    # rewards[f"agent_{i}"] += split
                    pass

        # Zone clearance bonus
        zone_bonus = self._check_zone_clearance() * self.rcfg["tier3_coverage"]["zone_clearance"] / self.n_agents
        for i in range(self.n_agents):
            # rewards[f"agent_{i}"] += zone_bonus
            pass

        # --- TIER 1: Victim detection (DOMINANT reward) ---
        for i in range(self.n_agents):
            key = f"agent_{i}"
            r, c = self.agent_positions[i]
            for v_idx, (vr, vc) in enumerate(self.victim_positions):
                dist = abs(r - vr) + abs(c - vc)
                thermal_val = self.thermal_map[r, c]
                if dist <= self.thermal_radius and thermal_val > 0.1:
                    if not self.victim_found[v_idx]:
                        self.victim_found[v_idx] = True
                        self.victim_detection_times[v_idx] = self.step_count
                        
                        gs = self.grid_size
                        corners = [(2, 2), (2, gs - 3), (gs - 3, 2), (gs - 3, gs - 3)]
                        dist_to_base = min([abs(vr - cr) + abs(vc - cc) for cr, cc in corners])
                        
                        with open("found_victims.log", "a") as f:
                            f.write(f"[GROUND TEAM ALERT] Victim {v_idx} found at (Row: {vr}, Col: {vc}) | Distance from nearest base: {dist_to_base} units | Found by Agent {i} at Step {self.step_count}\n")
                        
                        for dr in range(-5, 6):
                            for dc in range(-5, 6):
                                ntr, ntc = vr + dr, vc + dc
                                if 0 <= ntr < self.grid_size and 0 <= ntc < self.grid_size:
                                    self.thermal_map[ntr, ntc] = 0.0
                        # First detection bonus
                        # rewards[key] += self.rcfg["tier1_victim"]["first_detection"]
                        # Speed bonus: decays over episode
                        speed_factor = self.rcfg["tier1_victim"]["detection_speed_decay"] ** self.step_count
                        # rewards[key] += self.rcfg["tier1_victim"]["detection_speed_bonus_max"] * speed_factor
                        # Thermal confirmation if close enough
                        if dist <= 1:
                            # rewards[key] += self.rcfg["tier1_victim"]["thermal_confirmation"]
                            pass
                    else:
                        # Multi-agent corroboration bonus (second agent verifies)
                        already_corroborated = sum(1 for j in range(self.n_agents)
                                                    if j != i and
                                                    abs(self.agent_positions[j][0] - vr) +
                                                    abs(self.agent_positions[j][1] - vc) <= self.thermal_radius)
                        if already_corroborated >= 1:
                            # rewards[key] += self.rcfg["tier1_victim"]["multi_agent_corroboration"] * 0.1
                            pass

        # All victims found mega-bonus
        if self.victim_found.all() and not hasattr(self, "_all_found_rewarded"):
            self._all_found_rewarded = True
            split = self.rcfg["tier1_victim"]["all_victims_found"] / self.n_agents
            for i in range(self.n_agents):
                # rewards[f"agent_{i}"] += split
                pass

        # --- TIER 4: Team coordination ---
        spread_bonus = self._compute_spread_bonus()
        for i in range(self.n_agents):
            # rewards[f"agent_{i}"] += spread_bonus
            pass

        # --- TIER 5: Temporal ---
        for i in range(self.n_agents):
            # rewards[f"agent_{i}"] += self.rcfg["tier5_temporal"]["time_step_penalty"]
            if np.array_equal(self.agent_positions[i], list(prev_positions[i])):
                # rewards[f"agent_{i}"] += self.rcfg["tier5_temporal"]["hover_penalty"]
                pass
            self.agent_battery[i] -= 1.0

        # Mission complete bonus
        terminated = bool(self.victim_found.all())
        truncated = bool(self.step_count >= self.max_steps)
        if terminated:
            split = self.rcfg["tier5_temporal"]["mission_complete"] / self.n_agents
            for i in range(self.n_agents):
                # rewards[f"agent_{i}"] += split
                pass

        # End-of-episode missed victim penalty
        if truncated and not terminated:
            missed = int((~self.victim_found).sum())
            penalty = self.rcfg["tier1_victim"]["missed_victim_end_penalty"] * missed / self.n_agents
            for i in range(self.n_agents):
                # rewards[f"agent_{i}"] += penalty
                pass

        obs = self._build_observations()
        info = {
            "action_masks": self._get_all_action_masks(),
            "victims_found": int(self.victim_found.sum()),
            "total_collisions": int(self.collision_counts.sum()),
            "coverage_pct": float(coverage_pct),
            "step_count": self.step_count,
        }
        return obs, rewards, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _spawn_agents(self) -> np.ndarray:
        positions = []
        corners = [(2, 2), (2, self.grid_size - 3),
                   (self.grid_size - 3, 2), (self.grid_size - 3, self.grid_size - 3)]
        for i in range(self.n_agents):
            cr, cc = corners[i % len(corners)]
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = cr + dr, cc + dc
                    if (0 <= r < self.grid_size and 0 <= c < self.grid_size and
                            self.obstacle_map[r, c] == 0):
                        positions.append(np.array([r, c]))
                        break
                else:
                    continue
                break
            else:
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

    def _count_frontier_cells(self) -> int:
        count = 0
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                if self.coverage_map[r, c] > 0 and self.obstacle_map[r, c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if self.coverage_map[r + dr, c + dc] == 0 and self.obstacle_map[r + dr, c + dc] == 0:
                            count += 1
                            break
        return count

    def _check_zone_clearance(self) -> int:
        """Check how many quadrants are fully covered."""
        half = self.grid_size // 2
        zones_cleared = 0
        for (r0, r1), (c0, c1) in [((0, half), (0, half)), ((0, half), (half, self.grid_size)),
                                     ((half, self.grid_size), (0, half)),
                                     ((half, self.grid_size), (half, self.grid_size))]:
            zone = self.obstacle_map[r0:r1, c0:c1]
            passable = (zone == 0).sum()
            covered = (self.coverage_map[r0:r1, c0:c1][zone == 0] > 0).sum()
            if passable > 0 and covered / passable >= 0.9:
                zones_cleared += 1
        return zones_cleared

    def _compute_spread_bonus(self) -> float:
        if self.n_agents < 2:
            return 0.0
        positions = np.array(self.agent_positions, dtype=float)
        dists = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dists.append(np.linalg.norm(positions[i] - positions[j]))
        avg_dist = float(np.mean(dists))
        optimal_dist = self.grid_size * 0.4
        ratio = min(avg_dist / optimal_dist, 1.0)
        return self.rcfg["tier4_coordination"]["good_spread_bonus"] * ratio

    def _get_action_mask(self, agent_idx: int) -> np.ndarray:
        """
        Return boolean mask of valid actions for an agent.
        True = action is valid/allowed.
        """
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

        # Local occupancy grid (11x11)
        pad = self.obs_radius
        local = np.full((2 * pad + 1, 2 * pad + 1), -1.0, dtype=np.float32)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    local[dr + pad, dc + pad] = float(self.obstacle_map[nr, nc])

        # Thermal map (5x5)
        tp = self.thermal_radius
        thermal = np.zeros((2 * tp + 1, 2 * tp + 1), dtype=np.float32)
        for dr in range(-tp, tp + 1):
            for dc in range(-tp, tp + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    thermal[dr + tp, dc + tp] = self.thermal_map[nr, nc]

        # Normalized position
        pos = np.array([r / gs, c / gs], dtype=np.float32)

        # Relative positions of other agents
        other = []
        for j in range(self.n_agents):
            if j != agent_idx:
                or_, oc = self.agent_positions[j]
                other.extend([(or_ - r) / gs, (oc - c) / gs])
        other = np.array(other, dtype=np.float32)

        # Extras
        battery_ratio = self.agent_battery[agent_idx] / self.cfg["agent_battery_max"]
        step_ratio = self.step_count / self.max_steps
        victim_ratio = float(self.victim_found.sum()) / self.n_victims
        extras = np.array([battery_ratio, step_ratio, victim_ratio], dtype=np.float32)

        # Action mask (included in obs for network conditioning)
        mask = self._get_action_mask(agent_idx).astype(np.float32)

        return np.concatenate([local.flatten(), thermal.flatten(), pos, other, extras, mask])

    def _build_observations(self) -> Dict[str, np.ndarray]:
        return {f"agent_{i}": self._build_single_obs(i) for i in range(self.n_agents)}

    def get_global_state(self) -> np.ndarray:
        """For centralized critic: full global state vector."""
        parts = [self.obstacle_map.flatten().astype(np.float32),
                 self.coverage_map.flatten().astype(np.float32),
                 self.thermal_map.flatten().astype(np.float32),
                 self.victim_found.astype(np.float32)]
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            parts.append(np.array([r / self.grid_size, c / self.grid_size], dtype=np.float32))
        return np.concatenate(parts)

    @property
    def global_state_dim(self) -> int:
        return 3 * self.grid_size ** 2 + self.n_victims + self.n_agents * 2

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        display = np.zeros((self.grid_size, self.grid_size, 3))
        # Obstacles: dark gray
        display[self.obstacle_map == 1] = [0.3, 0.3, 0.3]
        # Coverage: light blue
        covered = self.coverage_map > 0
        display[covered & (self.obstacle_map == 0)] = [0.7, 0.85, 1.0]
        # Thermal overlay
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.thermal_map[r, c] > 0.3:
                    display[r, c] = [1.0, 0.5 * (1 - self.thermal_map[r, c]), 0]
        # Victims
        for idx, (vr, vc) in enumerate(self.victim_positions):
            if self.victim_found[idx]:
                display[vr, vc] = [0, 1, 0]
            else:
                display[vr, vc] = [1, 0, 0]
        ax.imshow(display, origin="upper")
        colors = ["cyan", "yellow", "magenta", "white"]
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            ax.scatter(c, r, color=colors[i % len(colors)], s=80, marker="^", zorder=5)
        ax.set_title(f"Step {self.step_count} | Victims: {self.victim_found.sum()}/{self.n_victims} | "
                     f"Coverage: {100*self.coverage_map[self.obstacle_map==0].mean():.1f}%")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
