import os
import json
import time
from datetime import datetime
from typing import Dict


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log = os.path.join(log_dir, f"run_{ts}.jsonl")
        self.eval_log = os.path.join(log_dir, f"eval_{ts}.jsonl")

    def log_epoch(self, metrics: Dict):
        metrics["timestamp"] = time.time()
        with open(self.run_log, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

    def log_eval(self, epoch: int, metrics: Dict):
        record = {"epoch": epoch, "timestamp": time.time(), **metrics}
        with open(self.eval_log, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def print_epoch(self, epoch: int, metrics: Dict):
        print(f"\n[Epoch {epoch:04d}] "
              f"Cov: {metrics.get('avg_coverage', 0):.1f}% | "
              f"Victims: {metrics.get('avg_victims_found', 0):.2f}/7 | "
              f"Coll: {metrics.get('avg_collisions', 0):.2f} | "
              f"SuccessRate: {metrics.get('success_rate', 0)*100:.1f}%")
