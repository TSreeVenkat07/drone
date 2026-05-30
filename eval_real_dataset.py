"""
Evaluation script on Real-World inspired datasets/layouts.
Benchmarks:
1. OSM City Grid (Manhattan/London Style dense blocks with collapsed ruins)
2. Wildfire Urban-Wildland Interface (Wind-driven fire perimeters and smoke plumes)
3. UN-SPIDER Flood Inundation Grid (Diagonal river overflow, flooded streets, debris blockages, and dry rooftop refuges)

Usage:
  python eval_real_dataset.py --checkpoint checkpoints/latest.pt --n_episodes 10
"""
import argparse
import os
import torch
import yaml
import numpy as np
from environment import DisasterEnv
from agents import UAVAgent
from baseline.greedy_agent import run_greedy_episode
from evaluation.metrics import MetricsTracker

class OSMBuildingCollapseGenerator:
    """Generates a dense urban city block resembling OpenStreetMap layout with collapsed building ruins."""
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def generate(self):
        # 0 = open, 1 = solid building / rubble
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Create a grid-based street system (OSM style)
        # Major avenues every 10 cells, minor streets every 5 cells
        avenues = [x for x in range(0, self.grid_size, 10)]
        streets = [y for y in range(0, self.grid_size, 5)]

        # Fill blocks with buildings
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                is_road = False
                for ave in avenues:
                    if abs(r - ave) <= 1:
                        is_road = True
                for st in streets:
                    if abs(c - st) <= 1:
                        is_road = True
                
                if not is_road:
                    # Place building blocks (with small gaps representing yards)
                    if (r % 4 != 0) and (c % 4 != 0):
                        grid[r, c] = 1.0

        # Simulate structural collapses (ruins) in OSM blocks
        # We turn 30% of building blocks into passable collapsed zones (rubble / rubble-filled streets)
        np.random.seed(42) # Replicable collapse pattern
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] == 1.0 and np.random.random() < 0.35:
                    grid[r, c] = 0.0 # Building collapsed, now a navigable gap with ruins

        return grid

class RealWildfireGenerator:
    """Generates a forest-residential boundary with wind-driven fire line and smoke plume overlays."""
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def generate(self):
        # 1. Base obstacle grid: Trees (passable) and houses (obstacles)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Right half is residential, left half is forest
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if c >= self.grid_size // 2:
                    # Residential house blocks
                    if (r % 3 == 0) and (c % 3 == 0):
                        grid[r, c] = 1.0

        # 2. Fire map & Thermal sources (Fire front moving from SW to NE)
        fire_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Diagonal fire front
                dist_front = (r - (self.grid_size - c))
                if -4 <= dist_front <= 4:
                    fire_map[r, c] = 1.0 # Active fire front (no fly zone)
                    grid[r, c] = 1.0 # Treat active fire as solid obstacle
                elif dist_front > 4:
                    fire_map[r, c] = 0.4 # Burned out / smoldering
                
                # Wind blows smoke to the North-East (top-right)
                # Smoke density increases near the fire line and trails off downwind
                if dist_front <= 8:
                    smoke_map[r, c] = max(0.0, 1.0 - abs(dist_front) / 10.0)

        self.fire_map = fire_map
        self.smoke_map = smoke_map
        return grid

class UNSpiderFloodGenerator:
    """Generates river inundation patterns, flooded streets, debris blockages, and dry rooftop refuges."""
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def generate(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 1. Base street/house grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r % 4 == 0) and (c % 4 == 0):
                    grid[r, c] = 1.0 # Buildings

        # 2. River overflow inundation mask (satellite-based radar representation)
        # River runs diagonally from bottom-left to top-right
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dist_to_river = abs(r - c)
                if dist_to_river <= 5:
                    # Inundated channel
                    if grid[r, c] == 0.0:
                        grid[r, c] = 0.5 # Flooded street (restricted speed / double battery)
                elif 5 < dist_to_river <= 10:
                    # Partial street flooding (random pools)
                    if grid[r, c] == 0.0 and (r + c) % 3 == 0:
                        grid[r, c] = 0.5
                
                # Stranded dry roofs: Keep buildings dry (value 0.0 for search landing, but cannot pass through unless searched)
                # In this case, we keep houses at 1.0 (blocked) unless they are high roofs (safe targets)
                # To represent safe rooftops, we clear them to 0.0 so drones can fly directly over them to check
                if grid[r, c] == 1.0 and dist_to_river <= 4 and (r * c) % 2 == 0:
                    grid[r, c] = 0.0 # Dry rooftop island target

        # 3. Random debris blockages blocking major streets
        np.random.seed(99)
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                if grid[r, c] == 0.5 and np.random.random() < 0.08:
                    grid[r, c] = 1.0 # Debris piles

        return grid

class RealDatasetEnvWrapper(DisasterEnv):
    """Subclasses DisasterEnv to inject real-world procedural scenarios."""
    def __init__(self, env_cfg_path, reward_cfg_path, scenario, n_agents, difficulty):
        super().__init__(env_cfg_path, reward_cfg_path, scenario, n_agents, difficulty)
        
        # Initialize generators
        self.osm_gen = OSMBuildingCollapseGenerator(self.grid_size)
        self.wildfire_gen = RealWildfireGenerator(self.grid_size)
        self.flood_gen = UNSpiderFloodGenerator(self.grid_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.coverage_milestone_achieved = set()
        self.victim_detection_times = {}

        # Override obstacle map based on scenario
        if self.scenario == "building_collapse":
            self.obstacle_map = self.osm_gen.generate()
            self.smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        elif self.scenario == "wildfire":
            self.obstacle_map = self.wildfire_gen.generate()
            self.smoke_map = self.wildfire_gen.smoke_map.copy()
            self.thermal_map = np.clip(self.thermal_map + self.wildfire_gen.fire_map, 0.0, 1.0)
        elif self.scenario == "flood":
            self.obstacle_map = self.flood_gen.generate()
            self.smoke_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Place victims in navigable open spaces/collapses
        diff_params = self.cfg["difficulties"][self.difficulty]
        self.victim_positions, thermal_base = self.victim_mgr.place_victims(
            self.obstacle_map, self.n_victims, diff_params["victim_visibility"]
        )
        self.thermal_map = np.clip(self.thermal_map + thermal_base, 0.0, 1.0)

        # Spawn agents in cleared corners
        self.agent_positions = self._spawn_agents()
        self.agent_battery = np.full(self.n_agents, self.cfg["agent_battery_max"], dtype=np.float32)
        self.collision_counts = np.zeros(self.n_agents, dtype=int)
        self._update_coverage()

        # Build initial observations
        obs = self._build_observations()
        info = {"action_masks": self._get_all_action_masks()}
        return obs, info

def run_real_evaluation(scenario_name, checkpoint_path, n_episodes, difficulty="medium"):
    print(f"\nEvaluating scenario: {scenario_name.upper()} ({n_episodes} episodes)...")
    
    # Load configuration files
    with open("config/agent_config.yaml") as f:
        acfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build probe environment to get dimensions
    probe_env = RealDatasetEnvWrapper("config/env_config.yaml", "config/reward_config.yaml", scenario_name, 4, difficulty)
    probe_env.reset()
    
    # Initialize agents
    agents = [UAVAgent(probe_env.obs_dim, probe_env.N_ACTIONS, acfg, str(device)) for _ in range(4)]
    
    # Load model weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    for i, agent in enumerate(agents):
        key = f"agent_{i}"
        if key in ckpt:
            agent.online_net.load_state_dict(ckpt[key]["online"])
            agent.epsilon = 0.0
            agent.online_net.eval()
            
    tracker = MetricsTracker()

    for ep_idx in range(n_episodes):
        # 1. Tabu Search VDN (Ours)
        env = RealDatasetEnvWrapper("config/env_config.yaml", "config/reward_config.yaml", scenario_name, 4, difficulty)
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

        # 2. Greedy Baseline
        env2 = RealDatasetEnvWrapper("config/env_config.yaml", "config/reward_config.yaml", scenario_name, 4, difficulty)
        g_metrics = run_greedy_episode(env2, 4)
        tracker.add_greedy_episode(g_metrics)

    summary = tracker.compute_summary()
    
    # Calculate greedy statistics explicitly to avoid tracker key omission
    summary["greedy_avg_collisions"] = float(np.mean([e["total_collisions"] for e in tracker.greedy_episodes]))
    summary["greedy_avg_victims_found"] = float(np.mean([e["victims_found"] for e in tracker.greedy_episodes]))
    summary["greedy_detection_accuracy"] = float(np.mean([e["victims_found"] / 7.0 for e in tracker.greedy_episodes]))
    
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--difficulty", default="medium")
    args = p.parse_args()

    scenarios = ["building_collapse", "wildfire", "flood"]
    results = {}

    print("=" * 80)
    print("REAL DISASTER DATASET EVALUATION BENCHMARK: TABU SEARCH VDN VS GREEDY BASELINE")
    print(f"Checkpoint: {args.checkpoint} | Episodes: {args.n_episodes} | Difficulty: {args.difficulty}")
    print("=" * 80)

    for scenario in scenarios:
        results[scenario] = run_real_evaluation(scenario, args.checkpoint, args.n_episodes, args.difficulty)

    # 1. Write the combined report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FINAL COMPARATIVE REPORT - REAL DATASET LAYOUTS")
    report_lines.append(f"Checkpoint: {args.checkpoint} | Episodes per scenario: {args.n_episodes}")
    report_lines.append("=" * 80)

    for scenario in scenarios:
        sum_data = results[scenario]
        
        report_lines.append(f"\nScenario: {scenario.upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"  Coverage:   Ours={sum_data['marl_coverage_pct']:.1f}% | Greedy={sum_data['greedy_coverage_pct']:.1f}% | Improvement: {sum_data['coverage_improvement_pct']:+.1f}%")
        report_lines.append(f"  Latency:    Ours={sum_data['marl_detection_step']:.1f} steps | Greedy={sum_data['greedy_detection_step']:.1f} steps | Reduction: {sum_data['latency_reduction_pct']:+.1f}%")
        report_lines.append(f"  Collisions: Ours={sum_data['avg_collisions']:.2f}/ep | Greedy={sum_data['greedy_avg_collisions']:.2f}/ep")
        report_lines.append(f"  Det. Acc:   Ours={sum_data['detection_accuracy']*100:.1f}% (Found: {sum_data['avg_victims_found']:.2f}/7) | Greedy={sum_data['greedy_detection_accuracy']*100:.1f}% (Found: {sum_data['greedy_avg_victims_found']:.2f}/7)")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("Summary of Goals Met across Real Benchmarks:")
    report_lines.append("=" * 80)
    for scenario in scenarios:
        sum_data = results[scenario]
        report_lines.append(f"  {scenario.upper()}:")
        for goal, met in sum_data.get("goals_met", {}).items():
            status = "YES - MET" if met else "NO - NOT MET"
            report_lines.append(f"    [{status}] {goal}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    output_filename = "real_dataset_eval_results.txt"
    with open(output_filename, "w") as f:
        f.write(report_text + "\n")
    print(f"\nCombined results successfully written to {output_filename}")

    # 2. Write individual scenario files
    for scenario in scenarios:
        sum_data = results[scenario]
        scen_lines = []
        scen_lines.append("=" * 80)
        scen_lines.append(f"REAL DATASET EVALUATION REPORT - {scenario.upper()}")
        scen_lines.append(f"Checkpoint: {args.checkpoint} | Episodes: {args.n_episodes}")
        scen_lines.append("=" * 80)
        
        scen_lines.append(f"\nResults for {scenario.upper()}:")
        scen_lines.append("-" * 40)
        scen_lines.append(f"  Coverage:   Ours={sum_data['marl_coverage_pct']:.1f}% | Greedy={sum_data['greedy_coverage_pct']:.1f}% | Improvement: {sum_data['coverage_improvement_pct']:+.1f}%")
        scen_lines.append(f"  Latency:    Ours={sum_data['marl_detection_step']:.1f} steps | Greedy={sum_data['greedy_detection_step']:.1f} steps | Reduction: {sum_data['latency_reduction_pct']:+.1f}%")
        scen_lines.append(f"  Collisions: Ours={sum_data['avg_collisions']:.2f}/ep | Greedy={sum_data['greedy_avg_collisions']:.2f}/ep")
        scen_lines.append(f"  Det. Acc:   Ours={sum_data['detection_accuracy']*100:.1f}% (Found: {sum_data['avg_victims_found']:.2f}/7) | Greedy={sum_data['greedy_detection_accuracy']*100:.1f}% (Found: {sum_data['greedy_avg_victims_found']:.2f}/7)")
        
        scen_lines.append("\nGoals Status:")
        for goal, met in sum_data.get("goals_met", {}).items():
            status = "YES - MET" if met else "NO - NOT MET"
            scen_lines.append(f"  [{status}] {goal}")
        
        scen_lines.append("=" * 80)
        
        scen_text = "\n".join(scen_lines)
        scen_filename = f"real_dataset_{scenario}_results.txt"
        with open(scen_filename, "w") as f:
            f.write(scen_text + "\n")
        print(f"Scenario results successfully written to {scen_filename}")

if __name__ == "__main__":
    main()
