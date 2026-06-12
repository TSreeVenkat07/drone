import os
import sys
import shutil
import subprocess

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "xbd_raw")
    selected_dir = os.path.join(base_dir, "data", "selected")

    print("=" * 80)
    print("STEP 1: Downloading & Selecting xBD Dataset")
    print("=" * 80)
    # This will trigger Kaggle download (needs credentials set up)
    run_command("python download_xbd.py")

    print("\n" + "=" * 80)
    print("STEP 2: Running 30-Episode Evaluation on Extracted Images")
    print("=" * 80)
    run_command("python run_evaluation.py --n_episodes 30")

    print("\n" + "=" * 80)
    print("STEP 3: Cleaning up downloaded 23GB dataset to free memory")
    print("=" * 80)
    if os.path.exists(raw_dir):
        print(f"Deleting huge raw data directory: {raw_dir}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("Raw data deleted successfully!")
    else:
        print(f"Directory {raw_dir} not found (already deleted?).")
        
    print("\nAll tasks completed. Results are saved in the results/ folder.")

if __name__ == "__main__":
    main()
