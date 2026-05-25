"""
Government-Grade 5-Tier Reward Engine
Hierarchy (highest to lowest priority):
  T1 MISSION-CRITICAL : Victim detection  (+80 to +500)
  T2 SAFETY-CRITICAL  : Collision avoidance (-50 to -80)
  T3 OPERATIONAL      : Coverage & efficiency (+1.5 to +200)
  T4 COORDINATION     : Team spread (+0.5 to +2.0)
  T5 TEMPORAL         : Time pressure (-0.05 per step)
"""


class RewardEngine:
    """Standalone reward calculator. DisasterEnv integrates this inline."""
    def __init__(self, rcfg: dict):
        self.t1 = rcfg["tier1_victim"]
        self.t2 = rcfg["tier2_safety"]
        self.t3 = rcfg["tier3_coverage"]
        self.t4 = rcfg["tier4_coordination"]
        self.t5 = rcfg["tier5_temporal"]

    def victim_detection_reward(self, is_first: bool, distance: int,
                                 step: int, max_steps: int) -> float:
        r = 0.0
        if is_first:
            r += self.t1["first_detection"]
            speed = self.t1["detection_speed_decay"] ** step
            r += self.t1["detection_speed_bonus_max"] * speed
            if distance <= 1:
                r += self.t1["thermal_confirmation"]
        else:
            r += self.t1["multi_agent_corroboration"]
        return r

    def collision_penalty(self, collision_type: str, count: int) -> float:
        base = {"wall": self.t2["wall_collision"],
                "agent": self.t2["agent_collision"],
                "oob": self.t2["out_of_bounds"]}.get(collision_type, -10.0)
        if count >= self.t2["repeated_collision_threshold"]:
            base *= self.t2["repeated_collision_multiplier"]
        return base

    def coverage_reward(self, new_cells: float, is_frontier: bool,
                         zone_clears: int) -> float:
        r = new_cells * self.t3["new_cell"]
        if is_frontier:
            r += self.t3["frontier_bonus"]
        r += zone_clears * self.t3["zone_clearance"]
        return r
