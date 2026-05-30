import time
import os
import winsound
import subprocess
import ctypes

def play_beeps():
    print("Epoch 40 completed and evaluated! Triggering visual popup...")
    # Pop up a Windows message box
    ctypes.windll.user32.MessageBoxW(0, "Epoch 40 Training & Evaluation is completely finished!", "UAV SAR Alert", 0x40 | 0x1)

def run_greedy_vs_us_eval():
    print("\nExecuting evaluation test (eval_greedy_vs_us.py) on checkpoints/epoch_0040.pt...")
    try:
        # Run python eval_greedy_vs_us.py checkpoints/epoch_0040.pt
        res = subprocess.run(
            ["python", "eval_greedy_vs_us.py", "checkpoints/epoch_0040.pt"],
            capture_output=True,
            text=True,
            check=True
        )
        print(res.stdout)
    except Exception as e:
        print(f"Error running evaluation: {e}")

def monitor():
    eval_file = "venkateval.txt"
    print(f"Monitoring '{eval_file}' for Epoch 40 completion and evaluation...")
    
    while True:
        if os.path.exists(eval_file):
            try:
                with open(eval_file, "r") as f:
                    content = f.read()
                if "Epoch: 40 (completed)" in content or "Routine Eval (Epoch 40)" in content:
                    play_beeps()
                    time.sleep(2)  # Give time for checkpoints to finish saving
                    run_greedy_vs_us_eval()
                    break
            except Exception as e:
                print(f"Error reading file: {e}")
        
        time.sleep(15)  # Check every 15 seconds

if __name__ == "__main__":
    monitor()
