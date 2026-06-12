import matplotlib.pyplot as plt
import os
import re

def plot_live_track(file_path: str, output_path: str):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    episodes = []
    coverages = []
    victims = []
    collisions = []
    steps = []

    # Regex to parse: Epoch 0/10 | Ep 8/100 | Scen: building_collapse | Cov: 95.7% | Vic: 5/7 | Col: 0 | Steps: 500
    pattern = re.compile(r"Ep (\d+)/\d+ .*? Cov: ([\d.]+)% .*? Vic: (\d+)/\d+ .*? Col: (\d+) .*? Steps: (\d+)")

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep_num = int(match.group(1))
                cov = float(match.group(2))
                vic = int(match.group(3))
                col = int(match.group(4))
                step = int(match.group(5))

                episodes.append(ep_num)
                coverages.append(cov)
                victims.append(vic)
                collisions.append(col)
                steps.append(step)

    if not episodes:
        print("No valid data found in live_track.txt yet.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Live Training Progress (Episode-by-Episode)", fontsize=16)

    # Coverage
    axes[0, 0].plot(episodes, coverages, color="blue", alpha=0.7)
    axes[0, 0].set_title("Coverage Rate (%)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("% Covered")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Victims Found
    axes[0, 1].plot(episodes, victims, color="green", alpha=0.7)
    axes[0, 1].axhline(y=7, color="r", linestyle="--", label="Target (7)")
    axes[0, 1].set_title("Victims Found (out of 7)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_yticks(range(0, 9))
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Collisions
    axes[1, 0].plot(episodes, collisions, color="red", alpha=0.7)
    axes[1, 0].set_title("Total Collisions")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Steps taken
    axes[1, 1].plot(episodes, steps, color="purple", alpha=0.7)
    axes[1, 1].axhline(y=500, color="r", linestyle="--", label="Max Steps (Timeout)")
    axes[1, 1].set_title("Steps to Completion")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved live training curves to {output_path}")

if __name__ == "__main__":
    plot_live_track("live_track.txt", "results/live_training_curves.png")
