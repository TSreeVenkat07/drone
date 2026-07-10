import time
import os
import subprocess
import sys

def kill_old_training():
    print("Stopping current training task (easy)...")
    # PowerShell command to kill the train.py process
    ps_cmd = (
        'Get-CimInstance Win32_Process | '
        'Where-Object { $_.CommandLine -like "*train.py*" -and $_.CommandLine -notlike "*run_overnight*" } | '
        'ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'
    )
    try:
        subprocess.run(["powershell", "-Command", ps_cmd], check=True)
        print("Successfully stopped existing train.py process.")
    except Exception as e:
        print(f"Error stopping process: {e}")

def run_evaluation():
    print("\nRunning evaluation on Epoch 40...")
    try:
        res = subprocess.run(
            ["python", "eval_greedy_vs_us.py", "checkpoints/epoch_0040.pt"],
            capture_output=True,
            text=True,
            check=True
        )
        # Save evaluation result to file
        with open("epoch_40_eval_results.txt", "w") as f:
            f.write(res.stdout)
        print("Epoch 40 evaluation completed and written to epoch_40_eval_results.txt.")
        print(res.stdout)
    except Exception as e:
        print(f"Error running evaluation script: {e}")

def monitor_and_orchestrate():
    # Phase 2: Kill current training process (in case any old instances are hanging)
    kill_old_training()
    time.sleep(5)

    # Phase 3: Start new training process on medium difficulty up to Epoch 42
    # Start epoch 32, total_epochs 43 (completing at 42)
    # Wrap in retry loop in case of transient CUDA/system crashes
    retries = 0
    max_retries = 5
    epoch_35_eval_done = False
    epoch_40_eval_done = False

    while retries < max_retries:
        print(f"\nPhase 3: Starting medium difficulty training from Epoch 43 to Epoch 45 (Attempt {retries+1}/{max_retries})...")
        train_proc = subprocess.Popen(
            ["python", "-u", "train.py", "--resume", "--difficulty", "medium", "--epochs", "46"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Monitor training output line-by-line
        while True:
            line = train_proc.stdout.readline()
            if not line:
                break
            print(line, end="", flush=True)

            # Stop training after Epoch 45 is completed and saved
            if "Saved replay buffer to checkpoint.pt" in line:
                if os.path.exists("checkpoints/epoch_0045.pt"):
                    print("\n[REQUESTED STOP] Epoch 45 checkpoint and replay buffer fully saved! Stopping training as requested.")
                    train_proc.terminate()
                    train_proc.wait()
                    print("Training stopped successfully after Epoch 45.")
                    sys.exit(0)

                # Check for Epoch 35 checkpoint to evaluate mid-training
                if not epoch_35_eval_done and os.path.exists("checkpoints/epoch_0035.pt"):
                    print("\nEpoch 35 checkpoint saved! Running mid-training evaluation...")
                    try:
                        res35 = subprocess.run(
                            ["python", "eval_greedy_vs_us.py", "checkpoints/epoch_0035.pt"],
                            capture_output=True, text=True, check=True
                        )
                        with open("epoch_35_eval_results.txt", "w") as f:
                            f.write(res35.stdout)
                        print("Epoch 35 evaluation completed and written to epoch_35_eval_results.txt.")
                    except Exception as e:
                        print(f"Error evaluating Epoch 35: {e}")
                    epoch_35_eval_done = True

            # Check if Epoch 40 eval is complete in venkateval.txt
            if not epoch_40_eval_done:
                if os.path.exists("venkateval.txt"):
                    try:
                        with open("venkateval.txt", "r") as f:
                            content = f.read()
                        if "Epoch: 40 (completed)" in content or "Routine Eval (Epoch 40)" in content:
                            print("\nEpoch 40 completed and evaluated in venkateval.txt!")
                            epoch_40_eval_done = True
                            # Run the custom head-to-head comparison
                            run_evaluation()
                    except Exception as e:
                        print(f"Error checking venkateval.txt: {e}")

        # Wait for the process to exit
        rc = train_proc.wait()
        if rc == 0:
            print("\nTraining completed successfully!")
            break
        else:
            retries += 1
            print(f"\n[ALERT] Training process exited with error code {rc}.")
            if retries < max_retries:
                print("Retrying in 15 seconds (resuming from latest checkpoint)...")
                time.sleep(15)
            else:
                print("Max retries reached. Overnight training stopped due to persistent errors.")
                sys.exit(rc)

    print("\nOvernight training completed successfully!")

if __name__ == "__main__":
    monitor_and_orchestrate()
