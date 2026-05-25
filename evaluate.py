"""
Evaluation and comparison script.
Usage:
  python evaluate.py --checkpoint checkpoints/latest.pt
  python evaluate.py --checkpoint checkpoints/epoch_0049.pt --n_episodes 50
"""
import argparse
import torch
import yaml
from environment import DisasterEnv
from agents import UAVAgent
from evaluation import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--scenario", default="all")
    args = p.parse_args()

    with open("config/agent_config.yaml") as f:
        acfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                             "building_collapse", 4, "easy")
    probe_env.reset()

    agents = [UAVAgent(probe_env.obs_dim, probe_env.N_ACTIONS, acfg, str(device)) for _ in range(4)]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    for i, agent in enumerate(agents):
        key = f"agent_{i}"
        if key in ckpt:
            agent.online_net.load_state_dict(ckpt[key]["online"])
            agent.epsilon = 0.0
    print(f"Loaded checkpoint: {args.checkpoint}")

    scenarios = (["building_collapse", "wildfire", "flood"]
                  if args.scenario == "all" else [args.scenario])
    evaluator = Evaluator("config/env_config.yaml", "config/reward_config.yaml", agents, device)
    results = evaluator.evaluate(args.difficulty, scenarios, n_eval_eps=args.n_episodes)
    print("\nFinal results:", results)


if __name__ == "__main__":
    main()
