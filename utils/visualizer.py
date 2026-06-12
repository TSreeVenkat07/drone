import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import List, Dict


def plot_training_curves(log_file: str, output_dir: str = "results"):
    """Load training log and plot key metrics over epochs."""
    import json
    os.makedirs(output_dir, exist_ok=True)

    epochs, coverages, victims, collisions, success = [], [], [], [], []
    with open(log_file) as f:
        for line in f:
            rec = json.loads(line.strip())
            epochs.append(rec["epoch"])
            coverages.append(rec.get("avg_coverage", 0))
            victims.append(rec.get("avg_victims_found", 0))
            collisions.append(rec.get("avg_collisions", 0))
            success.append(rec.get("success_rate", 0) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("UAV SAR Training Progress", fontsize=14)

    axes[0, 0].plot(epochs, coverages, "b-")
    axes[0, 0].axhline(y=np.mean(coverages) * 1.38, color="r", linestyle="--", label="+38% target")
    axes[0, 0].set_title("Coverage Rate (%)"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend()

    axes[0, 1].plot(epochs, victims, "g-")
    axes[0, 1].axhline(y=6.3, color="r", linestyle="--", label="90% target (6.3/7)")
    axes[0, 1].set_title("Avg Victims Found"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    axes[1, 0].plot(epochs, collisions, "r-")
    axes[1, 0].axhline(y=2, color="orange", linestyle="--", label="Max 2 target")
    axes[1, 0].set_title("Avg Collisions/Episode"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend()

    axes[1, 1].plot(epochs, success, "m-")
    axes[1, 1].set_title("Success Rate (%)"); axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved training curves to {out}")
