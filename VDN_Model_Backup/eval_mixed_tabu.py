import argparse
import torch
import yaml
import numpy as np
from environment import DisasterEnv
from agents import UAVAgent
from baseline.greedy_agent import run_greedy_episode
from evaluation.metrics import MetricsTracker

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--n_episodes_per_difficulty", type=int, default=20)
    p.add_argument("--scenario", default="building_collapse")
    p.add_argument("--history_len", type=int, default=4)
    args = p.parse_args()

    with open("config/agent_config.yaml") as f:
        acfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    probe_env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                             args.scenario, 4, "easy")
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
    print(f"Loaded checkpoint: {args.checkpoint} with Tabu History Len: {args.history_len}")

    trackers = {
        "easy": MetricsTracker(),
        "medium": MetricsTracker(),
        "hard": MetricsTracker(),
        "mixed": MetricsTracker()
    }
    
    episode_details = []
    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        print(f"\nEvaluating difficulty: {difficulty} ({args.n_episodes_per_difficulty} episodes)...")
        for ep_idx in range(args.n_episodes_per_difficulty):
            # 1. MARL Episode with Tabu History Wrapper
            env = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                              args.scenario, 4, difficulty)
            obs, info = env.reset()
            masks = info["action_masks"]
            done = False
            step = 0
            first_det = None

            victim_positions = list(env.victim_positions)
            
            # Initialize history
            pos_history = {f"agent_{i}": [] for i in range(4)}

            while not done:
                step += 1
                actions = {}
                for i in range(4):
                    key = f"agent_{i}"
                    r, c = env.agent_positions[i]
                    
                    obs_t = torch.FloatTensor(obs[key]).unsqueeze(0).to(device)
                    mask_t = torch.BoolTensor(masks[key]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q = agents[i].online_net(obs_t, mask_t).squeeze(0)
                    
                    # Sort actions by Q-value
                    sorted_actions = torch.argsort(q, descending=True).cpu().numpy()
                    
                    best_action = None
                    for action in sorted_actions:
                        # Skip masked/invalid actions
                        if q[action].item() < -1e8:
                            continue
                        
                        # Calculate potential next position
                        dr, dc = env.ACTION_DELTAS[action]
                        nr, nc = r + dr, c + dc
                        
                        # Avoid recently visited locations if alternative exists
                        if (nr, nc) in pos_history[key]:
                            continue
                        else:
                            best_action = action
                            break
                    
                    # Fallback: if all valid actions lead to visited positions, pick the highest Q-value action
                    if best_action is None:
                        for action in sorted_actions:
                            if q[action].item() > -1e8:
                                best_action = action
                                break
                    
                    # Log actual movement to history
                    dr, dc = env.ACTION_DELTAS[best_action]
                    nr, nc = r + dr, c + dc
                    pos_history[key].append((nr, nc))
                    if len(pos_history[key]) > args.history_len:
                        pos_history[key].pop(0)
                        
                    actions[key] = best_action

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
            
            trackers[difficulty].add_marl_episode(marl_metrics)
            trackers["mixed"].add_marl_episode(marl_metrics)

            # 2. Greedy Baseline Episode
            env2 = DisasterEnv("config/env_config.yaml", "config/reward_config.yaml",
                               args.scenario, 4, difficulty)
            g_metrics = run_greedy_episode(env2, 4)
            
            trackers[difficulty].add_greedy_episode(g_metrics)
            trackers["mixed"].add_greedy_episode(g_metrics)

            episode_details.append({
                "difficulty": difficulty,
                "episode_index": ep_idx,
                "victim_positions": victim_positions,
                "marl_victims_found": info["victims_found"],
                "marl_collisions": info["total_collisions"],
                "marl_coverage": info["coverage_pct"],
                "greedy_coverage": g_metrics["coverage_pct"],
                "greedy_collisions": g_metrics["total_collisions"]
            })

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"TABU WRAPPER EVALUATION REPORT (HISTORY={args.history_len}) - MARL vs GREEDY")
    report_lines.append(f"Checkpoint: {args.checkpoint}")
    report_lines.append("=" * 70)

    for diff in difficulties:
        sum_diff = trackers[diff].compute_summary()
        greedy_avg_col = float(np.mean([e["total_collisions"] for e in trackers[diff].greedy_episodes]))
        greedy_avg_vf = float(np.mean([e["victims_found"] for e in trackers[diff].greedy_episodes]))
        greedy_det_acc = float(np.mean([e["victims_found"] / 7.0 for e in trackers[diff].greedy_episodes]))
        report_lines.append(f"\n--- DIFFICULTY: {diff.upper()} ({args.n_episodes_per_difficulty} eps) ---")
        report_lines.append(f"  Coverage:   MARL={sum_diff['marl_coverage_pct']:.1f}% | Greedy={sum_diff['greedy_coverage_pct']:.1f}% | Improv: {sum_diff['coverage_improvement_pct']:+.1f}%")
        report_lines.append(f"  Latency:    MARL={sum_diff['marl_detection_step']:.1f} steps | Greedy={sum_diff['greedy_detection_step']:.1f} steps | Reduc: {sum_diff['latency_reduction_pct']:+.1f}%")
        report_lines.append(f"  Collisions: MARL={sum_diff['avg_collisions']:.2f}/ep | Greedy={greedy_avg_col:.2f}/ep")
        report_lines.append(f"  Det. Acc:   MARL={sum_diff['detection_accuracy']*100:.1f}% (Found: {sum_diff['avg_victims_found']:.2f}/7) | Greedy={greedy_det_acc*100:.1f}% (Found: {greedy_avg_vf:.2f}/7)")

    summary_mixed = trackers["mixed"].compute_summary()
    greedy_mixed_col = float(np.mean([e["total_collisions"] for e in trackers["mixed"].greedy_episodes]))
    greedy_mixed_vf = float(np.mean([e["victims_found"] for e in trackers["mixed"].greedy_episodes]))
    greedy_mixed_det_acc = float(np.mean([e["victims_found"] / 7.0 for e in trackers["mixed"].greedy_episodes]))
    report_lines.append("\n" + "=" * 70)
    report_lines.append(f"OVERALL COMBINED MIXED RESULTS (EASY + MEDIUM + HARD)")
    report_lines.append("=" * 70)
    report_lines.append(f"  Coverage:   MARL={summary_mixed['marl_coverage_pct']:.1f}% | Greedy={summary_mixed['greedy_coverage_pct']:.1f}% "
                        f"| Improvement: {summary_mixed['coverage_improvement_pct']:+.1f}%")
    report_lines.append(f"  Latency:    MARL={summary_mixed['marl_detection_step']:.1f}steps | Greedy={summary_mixed['greedy_detection_step']:.1f}steps "
                        f"| Reduction: {summary_mixed['latency_reduction_pct']:+.1f}%")
    report_lines.append(f"  Collisions: MARL={summary_mixed['avg_collisions']:.2f}/ep | Greedy={greedy_mixed_col:.2f}/ep")
    report_lines.append(f"  Det. acc:   MARL={summary_mixed['detection_accuracy']*100:.1f}% | Greedy={greedy_mixed_det_acc*100:.1f}% (Found MARL: {summary_mixed['avg_victims_found']:.2f}/7 | Greedy: {greedy_mixed_vf:.2f}/7)")
    report_lines.append("  Goals:")
    for goal, met in summary_mixed["goals_met"].items():
        status = "YES - MET" if met else "NO - NOT MET"
        report_lines.append(f"    [{status}] {goal}")
    report_lines.append(f"  ALL GOALS ACHIEVED: {'YES' if summary_mixed['all_goals_achieved'] else 'NO'}")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    output_filename = "mixed_eval_tabu_results.txt"
    with open(output_filename, "w") as f:
        f.write(report_text + "\n")
    print(f"\nResults written to {output_filename}")

if __name__ == "__main__":
    main()
