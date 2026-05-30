import torch
import numpy as np
import yaml
from training import CTDETrainer
from environment import DisasterEnv
from baseline.greedy_agent import run_greedy_episode

def main():
    # Load training config
    with open("config/training_config.yaml") as f:
        tcfg = yaml.safe_load(f)

    trainer = CTDETrainer("config/env_config.yaml", "config/agent_config.yaml", "config/training_config.yaml", "config/reward_config.yaml")

    # Load checkpoint
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    epoch = trainer.curriculum.load_checkpoint(trainer.agents, trainer.critic, ckpt_path)

    # Set evaluation mode
    for agent in trainer.agents:
        agent.online_net.eval()

    difficulty = sys.argv[2] if len(sys.argv) > 2 else trainer.curriculum.get_difficulty()
    scenarios = trainer.curriculum.get_scenarios_for_epoch()
    n_eval_eps = 20
    print(f"Evaluation Config - Difficulty: {difficulty}, Scenarios: {scenarios}")

    marl_covs, marl_vics, marl_cols, marl_steps = [], [], [], []
    greedy_covs, greedy_vics, greedy_cols, greedy_steps = [], [], [], []

    print("Running evaluation (20 episodes)...")
    for ep_idx in range(n_eval_eps):
        scenario = scenarios[ep_idx % len(scenarios)]
        
        # 1. MARL Episode
        env = DisasterEnv(trainer.env_cfg, trainer.reward_cfg, scenario, trainer.n_agents, difficulty)
        obs, info = env.reset()
        masks = info["action_masks"]
        done = False
        step = 0
        while not done:
            step += 1
            actions = {}
            for i in range(trainer.n_agents):
                key = f"agent_{i}"
                obs_t = torch.FloatTensor(obs[key]).unsqueeze(0).to(trainer.device)
                mask_t = torch.BoolTensor(masks[key]).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    q = trainer.agents[i].online_net(obs_t, mask_t)
                    action = int(q.argmax().item())
                actions[key] = action
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            masks = info["action_masks"]
            
        marl_covs.append(info["coverage_pct"])
        marl_vics.append(info["victims_found"])
        marl_cols.append(info["total_collisions"])
        marl_steps.append(step)

        # 2. Greedy Baseline Episode
        env2 = DisasterEnv(trainer.env_cfg, trainer.reward_cfg, scenario, trainer.n_agents, difficulty)
        g_metrics = run_greedy_episode(env2, trainer.n_agents)
        greedy_covs.append(g_metrics["coverage_pct"])
        greedy_vics.append(g_metrics["victims_found"])
        greedy_cols.append(g_metrics["total_collisions"])
        greedy_steps.append(g_metrics["steps"])

    print("\n" + "="*50)
    print(f"EVALUATION RESULT: US (MARL) VS GREEDY (EPOCH {epoch})")
    print("="*50)
    print(f"Metric             | US (MARL)       | Greedy Baseline")
    print(f"-------------------|-----------------|----------------")
    print(f"Avg Coverage       | {np.mean(marl_covs):.2f}%          | {np.mean(greedy_covs):.2f}%")
    print(f"Avg Victims Found  | {np.mean(marl_vics):.2f} / 7        | {np.mean(greedy_vics):.2f} / 7")
    print(f"Avg Collisions     | {np.mean(marl_cols):.2f}            | {np.mean(greedy_cols):.2f}")
    print(f"Avg Steps          | {np.mean(marl_steps):.1f}            | {np.mean(greedy_steps):.1f}")
    print("="*50)

if __name__ == "__main__":
    main()
