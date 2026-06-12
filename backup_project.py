import os
import shutil

backup_dir = "VDN_Model_Backup"
files_to_copy = [
    ("checkpoints/latest.pt", "latest.pt"),
    ("mixed_eval_results.txt", "mixed_eval_results.txt"),
    ("mixed_eval_tabu_results.txt", "mixed_eval_tabu_results.txt"),
    ("found_victims_gps.log", "found_victims_gps.log"),
    ("airsim_wrapper.py", "airsim_wrapper.py"),
    ("test_takeoff.py", "test_takeoff.py"),
    ("run_airsim_exploration.bat", "run_airsim_exploration.bat")
]

dirs_to_copy = [
    ("config", "config"),
    ("baseline", "baseline"),
    ("environment", "environment")
]

print(f"Creating backup directory: {backup_dir}...")
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Copy files
for src, dest in files_to_copy:
    if os.path.exists(src):
        dest_path = os.path.join(backup_dir, dest)
        shutil.copy2(src, dest_path)
        print(f"Backed up file: {src} -> {dest_path}")
    else:
        print(f"[WARNING] File not found: {src}")

# Copy directories
for src, dest in dirs_to_copy:
    if os.path.exists(src):
        dest_path = os.path.join(backup_dir, dest)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(src, dest_path)
        print(f"Backed up directory: {src} -> {dest_path}")
    else:
        print(f"[WARNING] Directory not found: {src}")

print("\n[SUCCESS] VDN model backup completed successfully!")
