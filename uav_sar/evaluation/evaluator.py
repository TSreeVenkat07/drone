import numpy as np
from typing import List, Dict
from environment import DisasterEnv
from baseline import GreedyAgent
from baseline.greedy_agent import run_greedy_episode
from .metrics import MetricsTracker


class Evaluator:
    """Runs MARL agents and greedy baseline head-to-head for fair comparison."""

    def __init__(self, env_cfg: str, reward_cfg: str, agents: List, device):
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.agents = agents
        self.device = device
        self.tracker = MetricsTracker()
        self.n_agents = len(agents)

    def evaluate(self, difficulty: str, scenarios: List[str],
                  n_eval_eps: int = 20) -> Dict:
        import torch
        self.tracker.reset()

        for ep_idx in range(n_eval_eps):
            scenario = scenarios[ep_idx % len(scenarios)]

            # MARL episode (no exploration - epsilon=0)
            env = DisasterEnv(self.env_cfg, self.reward_cfg, scenario,
                               self.n_agents, difficulty)
            obs, info = env.reset()
            masks = info["action_masks"]
            done = False
            step = 0
            first_det = None

            for agent in self.agents:
                agent.online_net.eval()

            while not done:
                step += 1
                actions = {}
                for i in range(self.n_agents):
                    key = f"agent_{i}"
                    actions[key] = self.agents[i].select_action(obs[key], masks[key], explore=False)

                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                masks = info["action_masks"]
                if info["victims_found"] > 0 and first_det is None:
                    first_det = step

            for agent in self.agents:
                agent.online_net.train()

            self.tracker.add_marl_episode({
                "victims_found": info["victims_found"],
                "total_collisions": info["total_collisions"],
                "coverage_pct": info["coverage_pct"],
                "detection_step": first_det or step,
                "success": info["victims_found"] == env.n_victims,
            })

            # Greedy baseline episode on same scenario/difficulty
            env2 = DisasterEnv(self.env_cfg, self.reward_cfg, scenario,
                                self.n_agents, difficulty)
            g_metrics = run_greedy_episode(env2, self.n_agents)
            self.tracker.add_greedy_episode(g_metrics)

        summary = self.tracker.compute_summary()
        self.tracker.print_report()
        return summary
