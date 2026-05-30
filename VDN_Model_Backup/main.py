"""Entry point — validates environment and runs a short smoke test."""
import numpy as np
from environment import DisasterEnv
from baseline.greedy_agent import run_greedy_episode


def smoke_test():
    print("=== UAV SAR Smoke Test ===")
    for scenario in ["building_collapse", "wildfire", "flood"]:
        for difficulty in ["easy", "medium", "hard"]:
            env = DisasterEnv(
                env_config_path="config/env_config.yaml",
                reward_config_path="config/reward_config.yaml",
                scenario=scenario,
                n_agents=4,
                difficulty=difficulty,
            )
            obs, info = env.reset()
            assert len(obs) == 4, "Expected 4 agent observations"
            assert "action_masks" in info
            masks = info["action_masks"]
            # Check action masks are working
            for i in range(4):
                mask = masks[f"agent_{i}"]
                assert mask.dtype == bool
                assert mask.any(), f"Agent {i} has no valid actions — action masking bug"

            # Step with random valid actions
            actions = {}
            for i in range(4):
                valid = np.where(masks[f"agent_{i}"])[0]
                actions[f"agent_{i}"] = int(np.random.choice(valid))
            next_obs, rewards, term, trunc, info2 = env.step(actions)
            print(f"  [{scenario:20s} | {difficulty:6s}] OK obs_dim={env.obs_dim} | "
                  f"victims={info2['victims_found']} | collisions={info2['total_collisions']}")

    # Greedy baseline comparison
    print("\n=== Greedy Baseline ===")
    env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                       "building_collapse", 4, "medium")
    g = run_greedy_episode(env, 4)
    print(f"  Greedy: coverage={g['coverage_pct']:.1f}% victims={g['victims_found']}/7 "
          f"collisions={g['total_collisions']} steps={g['steps']}")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    smoke_test()
