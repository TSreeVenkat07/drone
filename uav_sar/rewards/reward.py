import numpy as np
import yaml
from collections import defaultdict

REWARD_VERSION = "v3_full_spectrum"

class Reward:
    REWARD_VERSION = "v3_full_spectrum"

    def __init__(self, reward_cfg: str = "config/reward_config.yaml"):
        self.collision_counts = defaultdict(int)
        self.last_vectors = {}  # Track last movement vector for FEMA straight-line bonus
        self.milestones_achieved = set() # Track coverage milestones per episode
        with open(reward_cfg, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def on_episode_start(self):
        """Reset tracking state at the start of every episode."""
        self.collision_counts.clear()
        self.last_vectors.clear()
        self.milestones_achieved.clear()

    def compute_reward(self, agent_idx: int, state: dict, action: int, next_state: dict) -> float:
        """
        Computes the v3_full_spectrum reward using values from reward_config.yaml.
        """
        total_reward = 0.0

        grid_size = state["grid_size"]
        obstacle_map = state["obstacle_map"]
        ar, ac = next_state["agent_positions"][agent_idx]
        prev_r, prev_c = state["agent_positions"][agent_idx]

        # Action logic
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
        
        action = int(action)
        dr, dc = ACTION_DELTAS[action]
        nr, nc = prev_r + dr, prev_c + dc
        
        # Collisions
        is_col = False
        if action != 8:
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                is_col = True
            elif obstacle_map[nr, nc] == 1:
                is_col = True
                
        # Agent-agent collision check
        for other_idx in next_state["agent_positions"]:
            if other_idx != agent_idx:
                if np.array_equal(next_state["agent_positions"][agent_idx], next_state["agent_positions"][other_idx]):
                    is_col = True

        if is_col:
            self.collision_counts[agent_idx] += 1
            count = min(self.collision_counts[agent_idx], self.cfg["tier2_safety"]["repeated_collision_threshold"] * 3)
            base_col_penalty = self.cfg["tier2_safety"]["wall_collision"]
            if self.collision_counts[agent_idx] >= self.cfg["tier2_safety"]["repeated_collision_threshold"]:
                base_col_penalty *= self.cfg["tier2_safety"]["repeated_collision_multiplier"] ** (count - self.cfg["tier2_safety"]["repeated_collision_threshold"] + 1)
            total_reward += base_col_penalty

        # TIER 1: Victim found
        newly_found = [
            v_idx for v_idx in range(len(state["victim_found"]))
            if not state["victim_found"][v_idx] and next_state["victim_found"][v_idx]
        ]
        
        for v_idx in newly_found:
            vr, vc = state["victim_positions"][v_idx]
            dist = abs(ar - vr) + abs(ac - vc)
            if dist <= state["thermal_radius"] and state["thermal_map"][ar, ac] > 0.1:
                total_reward += self.cfg["tier1_victim"]["first_detection"]
                # NEW: Detection speed bonus
                speed_factor = self.cfg["tier1_victim"]["detection_speed_decay"] ** state.get("step_count", 0)
                total_reward += self.cfg["tier1_victim"]["detection_speed_bonus_max"] * speed_factor
                # NEW: Thermal confirmation
                if dist <= 1:
                    total_reward += self.cfg["tier1_victim"]["thermal_confirmation"]

        # NEW EXCELLENT FIX: Thermal Gradient Homing (Hotter/Colder)
        current_heat = next_state["thermal_map"][ar, ac]
        prev_heat = state["thermal_map"][prev_r, prev_c]
        if current_heat > prev_heat and current_heat > 0.1:
            total_reward += (current_heat - prev_heat) * 50.0  # Big reward for following heat

        # NEW: All victims found mega bonus
        if not all(state["victim_found"]) and all(next_state["victim_found"]):
            total_reward += self.cfg["tier1_victim"]["all_victims_found"] / len(state["agent_positions"])

        # TIER 3: Coverage
        if state["coverage_map"][ar, ac] == 0:
            total_reward += self.cfg["tier3_coverage"]["new_cell"]
            
            # NEW: Deep Exploration Bonus (Anti-Fear)
            wall_neighbors = 0
            for dr_off, dc_off in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                fr, fc = ar + dr_off, ac + dc_off
                if 0 <= fr < grid_size and 0 <= fc < grid_size:
                    if obstacle_map[fr, fc] == 1:
                        wall_neighbors += 1
                else:
                    wall_neighbors += 1 # out of bounds counts as wall
            if wall_neighbors >= 5: # Dead end or deep corner
                total_reward += self.cfg.get("tier3_coverage", {}).get("deep_exploration_bonus", 5.0)
        
        # NEW: Redundant cell penalty
        if state["coverage_map"][ar, ac] > 0 and action != 8:
            total_reward += self.cfg.get("tier3_coverage", {}).get("redundant_penalty", 0.0)

        # NEW: Frontier Bonus
        unexplored_neighbors = 0
        for dr_off, dc_off in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            fr, fc = ar + dr_off, ac + dc_off
            if 0 <= fr < grid_size and 0 <= fc < grid_size:
                if state["coverage_map"][fr, fc] == 0 and obstacle_map[fr, fc] == 0:
                    unexplored_neighbors += 1
        if unexplored_neighbors > 0 and not is_col and action != 8:
            total_reward += self.cfg["tier3_coverage"]["frontier_bonus"] * (unexplored_neighbors / 8.0)

        # NEW: Coverage Milestones
        coverage_pct = np.sum(next_state["coverage_map"] > 0) / (grid_size ** 2) * 100
        for threshold, key in [(25,"milestone_25pct"),(50,"milestone_50pct"),
                                (75,"milestone_75pct"),(100,"milestone_100pct")]:
            if coverage_pct >= threshold and threshold not in self.milestones_achieved:
                self.milestones_achieved.add(threshold)
                total_reward += self.cfg["tier3_coverage"][key] / len(state["agent_positions"])

        # NEW: Coordination Spread and Traffic Yield
        min_dist = float('inf')
        for other_idx in next_state["agent_positions"]:
            if other_idx != agent_idx:
                opos = next_state["agent_positions"][other_idx]
                d = abs(ar - opos[0]) + abs(ac - opos[1])
                min_dist = min(min_dist, d)
                
                # Traffic Yield Bonus (Anti-Fear)
                prev_opos = state["agent_positions"][other_idx]
                prev_dist = abs(prev_r - prev_opos[0]) + abs(prev_c - prev_opos[1])
                if prev_dist <= 2:
                    if d > prev_dist or (action == 8 and d == prev_dist):
                        total_reward += self.cfg.get("tier4_coordination", {}).get("traffic_yield_bonus", 5.0)

        if min_dist > 5:
            total_reward += self.cfg.get("tier4_coordination", {}).get("good_spread_bonus", 0.0)
        elif min_dist <= 1:
            total_reward += self.cfg.get("tier4_coordination", {}).get("overlap_penalty", 0.0)

        # Temporal Penalties
        total_reward += self.cfg["tier5_temporal"]["time_step_penalty"]
        if action == 8: # Hovering
            total_reward += self.cfg["tier5_temporal"]["hover_penalty"]

        # FEMA TIER 6
        if "tier6_fema" in self.cfg:
            # 1. Wall Following
            near_wall = False
            for dr_off in [-1, 0, 1]:
                for dc_off in [-1, 0, 1]:
                    if dr_off == 0 and dc_off == 0:
                        continue
                    check_r, check_c = ar + dr_off, ac + dc_off
                    if not (0 <= check_r < grid_size and 0 <= check_c < grid_size):
                        near_wall = True
                        break
                    elif obstacle_map[check_r, check_c] == 1:
                        near_wall = True
                        break
                if near_wall:
                    break
            
            if near_wall and not is_col and action != 8:
                total_reward += self.cfg["tier6_fema"]["wall_follow_bonus"]

            # 2. Straight-Line Sector Sweep
            current_vector = (dr, dc)
            if action != 8:
                if agent_idx in self.last_vectors:
                    if self.last_vectors[agent_idx] == current_vector:
                        total_reward += self.cfg["tier6_fema"]["straight_line_bonus"]
                self.last_vectors[agent_idx] = current_vector

        return float(total_reward)

    # =========================================================================
    # PREVIOUS REWARD SYSTEM (COMMENTED OUT TO PREVENT USAGE)
    # =========================================================================
    # def compute_reward_v2(self, agent_idx: int, state: dict, action: int, next_state: dict) -> float:
    #     ... (v2_safe_navigation was strictly limiting exploration without frontier bonuses) ...
