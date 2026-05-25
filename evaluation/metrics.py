import numpy as np
from typing import List, Dict


class MetricsTracker:
    """
    Tracks all performance metrics and computes improvement vs greedy baseline.
    Target goals:
      - Coverage rate: +38% over greedy
      - Victim detection latency: −27% over greedy
      - Collisions per episode: 0–2 (Hard: ≤2, Medium: ≤1, Easy: 0)
      - Victim detection accuracy: ≥90% of all 7 victims found
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.marl_episodes: List[Dict] = []
        self.greedy_episodes: List[Dict] = []

    def add_marl_episode(self, metrics: Dict):
        self.marl_episodes.append(metrics)

    def add_greedy_episode(self, metrics: Dict):
        self.greedy_episodes.append(metrics)

    def compute_summary(self) -> Dict:
        if not self.marl_episodes or not self.greedy_episodes:
            return {}

        def avg(eps, key):
            return float(np.mean([e[key] for e in eps]))

        marl_cov = avg(self.marl_episodes, "coverage_pct")
        greedy_cov = avg(self.greedy_episodes, "coverage_pct")
        marl_lat = avg(self.marl_episodes, "detection_step")
        greedy_lat = avg(self.greedy_episodes, "detection_step")
        marl_col = avg(self.marl_episodes, "total_collisions")
        marl_vf = avg(self.marl_episodes, "victims_found")
        detect_rate = float(np.mean([e["victims_found"] / 7 for e in self.marl_episodes]))

        coverage_improvement = (marl_cov - greedy_cov) / max(greedy_cov, 1e-8) * 100
        latency_reduction = (greedy_lat - marl_lat) / max(greedy_lat, 1e-8) * 100

        goals_met = {
            "coverage_+38pct": coverage_improvement >= 38.0,
            "latency_-27pct": latency_reduction >= 27.0,
            "collisions_0to2": marl_col <= 2.0,
            "detection_accuracy_90pct": detect_rate >= 0.90,
        }

        return {
            "marl_coverage_pct": marl_cov,
            "greedy_coverage_pct": greedy_cov,
            "coverage_improvement_pct": coverage_improvement,
            "marl_detection_step": marl_lat,
            "greedy_detection_step": greedy_lat,
            "latency_reduction_pct": latency_reduction,
            "avg_collisions": marl_col,
            "avg_victims_found": marl_vf,
            "detection_accuracy": detect_rate,
            "goals_met": goals_met,
            "all_goals_achieved": all(goals_met.values()),
        }

    def print_report(self):
        s = self.compute_summary()
        if not s:
            print("No data yet.")
            return
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT vs GREEDY BASELINE")
        print("=" * 60)
        print(f"  Coverage:   MARL={s['marl_coverage_pct']:.1f}% | Greedy={s['greedy_coverage_pct']:.1f}% "
              f"| Improvement: {s['coverage_improvement_pct']:+.1f}% (target: +38%)")
        print(f"  Latency:    MARL={s['marl_detection_step']:.1f}steps | Greedy={s['greedy_detection_step']:.1f}steps "
              f"| Reduction: {s['latency_reduction_pct']:+.1f}% (target: -27%)")
        print(f"  Collisions: {s['avg_collisions']:.2f}/ep (target: 0-2)")
        print(f"  Det. acc:   {s['detection_accuracy']*100:.1f}% (target: >=90%)")
        print("  Goals:")
        for goal, met in s["goals_met"].items():
            status = "YES - MET" if met else "NO - NOT MET"
            print(f"    [{status}] {goal}")
        print(f"  ALL GOALS ACHIEVED: {'YES' if s['all_goals_achieved'] else 'NO'}")
        print("=" * 60)
