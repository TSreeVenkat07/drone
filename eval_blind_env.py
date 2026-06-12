import argparse
import torch
import yaml
import numpy as np
import random
from environment import DisasterEnv
from agents import UAVAgent
from baseline.greedy_agent import run_greedy_episode
from evaluation.metrics import MetricsTracker

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--n_episodes", type=int, default=30)
    args = p.parse_args()

    with open("config/agent_config.yaml") as f:
        acfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # Instantiate probe env just for dimensions
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
            agent.online_net.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    tracker = MetricsTracker()
    episode_details = []

    scenarios = ["building_collapse", "wildfire", "flood"]
    difficulties = ["easy", "medium", "hard"]

    print(f"\nEvaluating BLIND ENVIRONMENTS for {args.n_episodes} episodes...")
    
    for ep_idx in range(args.n_episodes):
        scenario = random.choice(scenarios)
        difficulty = random.choice(difficulties)
        
        # MARL Episode
        env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                          scenario, 4, difficulty)
        obs, info = env.reset()
        masks = info["action_masks"]
        done = False
        step = 0
        first_det = None
        
        while not done:
            step += 1
            actions = {}
            for i in range(4):
                key = f"agent_{i}"
                actions[key] = agents[i].select_action(obs[key], masks[key], explore=False)

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            masks = info["action_masks"]
            if info["victims_found"] > 0 and first_det is None:
                first_det = step

        marl_metrics = {
            "victims_found": info["victims_found"],
            "total_collisions": info["total_collisions"],
            "coverage_pct": info["coverage_pct"],
            "detection_step": first_det or step,
            "success": info["victims_found"] == env.n_victims,
        }
        
        tracker.add_marl_episode(marl_metrics)

        # Greedy Baseline Episode for comparison
        env2 = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                           scenario, 4, difficulty)
        g_metrics = run_greedy_episode(env2, 4)
        tracker.add_greedy_episode(g_metrics)

        episode_details.append({
            "episode_index": ep_idx,
            "scenario": scenario,
            "difficulty": difficulty,
            "marl_victims_found": info["victims_found"],
            "marl_collisions": info["total_collisions"],
            "marl_coverage": info["coverage_pct"],
            "greedy_coverage": g_metrics["coverage_pct"],
            "greedy_collisions": g_metrics["total_collisions"]
        })

    # Generate final report string
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL EVALUATION REPORT - BLIND ENVIRONMENT (RANDOM SCENARIOS)")
    report_lines.append(f"Checkpoint: {args.checkpoint}")
    report_lines.append("=" * 70)
    
    summary = tracker.compute_summary()
    greedy_avg_col = float(np.mean([e["total_collisions"] for e in tracker.greedy_episodes]))
    
    report_lines.append(f"OVERALL COMBINED MIXED RESULTS ({args.n_episodes} eps)")
    report_lines.append("=" * 70)
    report_lines.append(f"  Coverage:   MARL={summary['marl_coverage_pct']:.1f}% | Greedy={summary['greedy_coverage_pct']:.1f}% | Improv: {summary['coverage_improvement_pct']:+.1f}%")
    report_lines.append(f"  Latency:    MARL={summary['marl_detection_step']:.1f} steps | Greedy={summary['greedy_detection_step']:.1f} steps | Reduc: {summary['latency_reduction_pct']:+.1f}%")
    report_lines.append(f"  Collisions: MARL={summary['avg_collisions']:.2f}/ep | Greedy={greedy_avg_col:.2f}/ep")
    report_lines.append(f"  Det. Acc:   {summary['detection_accuracy']*100:.1f}% (Found: {summary['avg_victims_found']:.2f})")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("DETAILED EPISODE LOG")
    report_lines.append("=" * 70)
    for detail in episode_details:
        report_lines.append(
            f"Ep {detail['episode_index']:>2} | Env: {detail['scenario']:<18} | Diff: {detail['difficulty']:<7} | "
            f"MARL Found: {detail['marl_victims_found']} | "
            f"MARL Col: {detail['marl_collisions']:>2} | MARL Cov: {detail['marl_coverage']:.1f}% | "
            f"Greedy Cov: {detail['greedy_coverage']:.1f}%"
        )
    
    report_text = "\n".join(report_lines)
    print(report_text)

    output_filename = "blind_env_evaluation_results.txt"
    with open(output_filename, "w") as f:
        f.write(report_text + "\n")
    print(f"\nResults successfully written to {output_filename}")

if __name__ == "__main__":
    main()
