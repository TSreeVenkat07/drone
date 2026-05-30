import time
import os
import subprocess
import ctypes

def play_popup():
    print("Epoch 43 completed! Triggering silent visual popup...")
    # Pop up a Windows message box (silent)
    ctypes.windll.user32.MessageBoxW(0, "Epoch 43 Medium Perfection Training is completely finished!", "UAV SAR Alert", 0x40 | 0x1)

def run_greedy_vs_us_eval():
    print("\nExecuting final evaluation test (eval_greedy_vs_us.py) on checkpoints/epoch_0043.pt...")
    try:
        # Run python eval_greedy_vs_us.py checkpoints/epoch_0043.pt
        res = subprocess.run(
            ["python", "eval_greedy_vs_us.py", "checkpoints/epoch_0043.pt"],
            capture_output=True,
            text=True,
            check=True
        )
        print(res.stdout)
        
        # Save evaluation result to file for reference
        with open("epoch_43_eval_results.txt", "w") as f:
            f.write(res.stdout)
            
    except Exception as e:
        print(f"Error running evaluation: {e}")

def monitor():
    ckpt_file = "checkpoints/epoch_0043.pt"
    print(f"Monitoring '{ckpt_file}' for Epoch 43 completion...")
    
    while True:
        if os.path.exists(ckpt_file):
            # Check if file size is stable (meaning it finished saving)
            size1 = os.path.getsize(ckpt_file)
            time.sleep(3)
            size2 = os.path.getsize(ckpt_file)
            
            if size1 == size2 and size1 > 1000000000: # Ensure it's the full 1.7GB file
                print(f"Checkpoint fully saved ({size1} bytes). Triggering alert and eval...")
                play_popup()
                run_greedy_vs_us_eval()
                break
        
        time.sleep(15)  # Check every 15 seconds

if __name__ == "__main__":
    monitor()
