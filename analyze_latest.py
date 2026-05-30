import re
import numpy as np

def main():
    filepath = "c:/Users/sreev/Desktop/final/uav_sar/live_track.txt"
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+Ep\s+(\d+)/\d+\s+\|\s+Scen:\s+(\w+)\s+\|\s+Cov:\s+([\d.]+)%\s+\|\s+Vic:\s+(\d+)/7\s+\|\s+Col:\s+(\d+)\s+\|\s+Steps:\s+(\d+)"
    )
    
    # Read all lines
    lines_parsed = []
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            ep_idx = int(match.group(2))
            scen = match.group(3)
            cov = float(match.group(4))
            vic = int(match.group(5))
            col = int(match.group(6))
            steps = int(match.group(7))
            lines_parsed.append((epoch, ep_idx, scen, cov, vic, col, steps))
            
    # Find restarts/runs
    runs = []
    current_run = []
    last_ep_idx = -1
    last_epoch = -1
    for item in lines_parsed:
        epoch, ep_idx = item[0], item[1]
        if epoch != last_epoch or ep_idx <= last_ep_idx:
            if current_run:
                runs.append(current_run)
                current_run = []
        current_run.append(item)
        last_ep_idx = ep_idx
        last_epoch = epoch
    if current_run:
        runs.append(current_run)
        
    print(f"Total runs/phases detected: {len(runs)}")
    
    # Run 55 is the old Epoch 26 (before update)
    # Runs 56, 57, 58... are after update
    old_run = runs[55]
    old_covs = [x[3] for x in old_run]
    old_vics = [x[4] for x in old_run]
    old_cols = [x[5] for x in old_run]
    old_steps = [x[6] for x in old_run]
    
    print("\n### DETAILED PERFORMANCE EVOLUTION (BEFORE VS AFTER REWARD UPDATE)")
    print("| Run / Epoch | Status | Episodes | Avg Coverage (%) | Avg Victims Found | Avg Collisions | Avg Steps |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    print(f"| **Old Epoch 26** | Baseline (Old Reward) | {len(old_run)} | {np.mean(old_covs):.2f}% | {np.mean(old_vics):.2f} / 7 | {np.mean(old_cols):.2f} | {np.mean(old_steps):.1f} |")
    
    for idx in range(56, len(runs)):
        run = runs[idx]
        epoch_num = run[0][0]
        covs = [x[3] for x in run]
        vics = [x[4] for x in run]
        cols = [x[5] for x in run]
        steps = [x[6] for x in run]
        status = "Completed" if len(run) == 100 else "In Progress..."
        print(f"| **Epoch {epoch_num}** | New Reward ({status}) | {len(run)} | {np.mean(covs):.2f}% | {np.mean(vics):.2f} / 7 | {np.mean(cols):.2f} | {np.mean(steps):.1f} |")

if __name__ == "__main__":
    main()
