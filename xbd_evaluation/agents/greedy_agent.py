import numpy as np
from typing import List, Tuple, Dict


class GreedyAgent:
    """Greedy frontier-based baseline agent."""

    def __init__(self, agent_id: int, n_actions: int = 9):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.action_deltas = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)
        ]

    def select_action(self, obs, action_mask, coverage_map, obstacle_map, current_pos):
        r, c = current_pos
        gs = coverage_map.shape[0]
        best_dist = float("inf")
        best_action = 8

        for a, (dr, dc) in enumerate(self.action_deltas[:8]):
            if not action_mask[a]:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < gs and 0 <= nc < gs):
                continue
            if obstacle_map[nr, nc] == 1:
                continue
            if coverage_map[nr, nc] == 0:
                dist = 0.0
            else:
                dist = float(coverage_map[nr, nc])
            if dist < best_dist:
                best_dist = dist
                best_action = a

        return best_action


def run_greedy_episode(env, n_agents: int) -> Dict:
    greedy_agents = [GreedyAgent(i) for i in range(n_agents)]
    obs, info = env.reset()
    masks = info["action_masks"]
    done = False
    step = 0
    first_detection_step = None

    while not done:
        step += 1
        actions = {}
        for i in range(n_agents):
            key = f"agent_{i}"
            action = greedy_agents[i].select_action(
                obs[key], masks[key],
                env.coverage_map, env.obstacle_map,
                tuple(env.agent_positions[i])
            )
            actions[key] = action

        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        masks = info["action_masks"]

        if info["victims_found"] > 0 and first_detection_step is None:
            first_detection_step = step

    return {
        "victims_found": info["victims_found"],
        "total_collisions": info["total_collisions"],
        "coverage_pct": info["coverage_pct"],
        "steps": step,
        "detection_step": first_detection_step or step,
        "success": info["victims_found"] == env.n_victims,
    }
