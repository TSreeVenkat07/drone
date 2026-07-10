import time
import os
import subprocess
import torch

def main():
    print("Advancement evaluation manager started. Monitoring checkpoints...")
    
    target_epoch = 109
    checkpoint_path = os.path.join("checkpoints", "latest.pt")
    
    while True:
        if os.path.exists(checkpoint_path):
            try:
                # Load metadata only to be fast and safe
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                current_epoch = ckpt.get("epoch", 0)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current epoch: {current_epoch}/{target_epoch}")
                if current_epoch >= target_epoch:
                    print("Training target epoch reached! Starting mixed evaluation...")
                    break
            except Exception as e:
                # Checkpoint might be in the middle of being written, wait and retry
                print(f"Checkpoint check paused (writing in progress): {e}")
        time.sleep(45)
        
    # Wait a few seconds to ensure checkpoint is completely written
    time.sleep(5)
    
    # Run the comprehensive mixed evaluation with 20 episodes per difficulty
    print("Launching eval_mixed_tabu.py for 20 episodes per difficulty...")
    cmd = "python eval_mixed_tabu.py --n_episodes_per_difficulty 20 --history_len 4"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print("Evaluation execution completed!")
    print(res.stdout)
    if res.stderr:
        print("Error log:")
        print(res.stderr)

if __name__ == "__main__":
    main()
