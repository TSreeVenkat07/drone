# Multi-UAV Cooperative Search & Rescue (SAR) using VDN in 3D AirSim

This directory contains the final trained model weights, configuration parameters, source code wrappers, and evaluation results for our Multi-Agent Reinforcement Learning (MARL) Search and Rescue system. 

---

## 1. Project Architecture & Methodology

Our system uses a **Value-Decomposition Network (VDN)** architecture combined with a hybrid **Tabu Search** mechanism to control a team of 4 autonomous UAVs exploring disaster zones (specifically, `building_collapse` scenarios with high rubble/debris density).

### Core Components
*   **MARL Decision Policy (VDN)**: In a cooperative setting, VDN factorizes the joint team action-value $Q_{total}(\mathbf{s}, \mathbf{a})$ as a linear summation of individual utilities $Q_i(o_i, a_i)$ for each UAV $i$:
    $$Q_{total}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^{N} Q_i(o_i, a_i)$$
    This decomposition enables decentralized execution (each drone acts on its local observations $o_i$) based on centralized training.
*   **3D Simulator Integration (AirSim)**: We link the trained 2D policy to a high-fidelity 3D Unreal Engine simulator through a custom Python wrapper. Drones navigate a 32x32 grid (scaled at 5.0m per grid cell) at a safety altitude of $Z = -3.0$ meters.
*   **Sensor Emulation**:
    *   *LiDAR*: Projects point-cloud distances into a local 2D occupancy map to detect obstacles and clear space.
    *   *Thermal Sensors*: Scans localized areas for heat signatures of victims.
*   **Georeferencing (WGS84)**: Converts simulator local grid offsets into absolute GPS Latitude/Longitude coordinates and computes physical rescue distances to the nearest ground team base.
*   **Safety Overrides**: An autopilot proximity-override triggers at 10Hz to automatically hover a drone if it comes within a threshold range of obstacles or other UAVs to prevent collisions.

---

## 2. Final Evaluation Performance Metrics

We evaluated the VDN model (`latest.pt`, Epoch 109) across Easy, Medium, and Hard terrains. We compared the **Raw MARL VDN** policy and our **Hybrid VDN + Tabu Search (History = 4)** policy against a **Greedy Baseline**.

### A. Hybrid VDN + Tabu Search vs. Greedy Baseline
The Hybrid policy adds a history-based Tabu memory to prevent drones from repeating exploratory trajectories, resulting in massive efficiency gains:

| Metric | Hybrid VDN + Tabu | Greedy Baseline | Improvement / Comparison |
| :--- | :---: | :---: | :---: |
| **Average Coverage (All Difficulties)** | **76.5%** | 59.6% | **+28.2% Coverage Increase** |
| **Exploration Latency (All Difficulties)** | **20.8 steps** | 39.6 steps | **+47.6% Latency Reduction** |
| **Collisions per Episode (All Difficulties)**| **5.60** | 1278.70 | **99.5% Collision Reduction** |
| **Victim Detection Accuracy** | **41.2%** (2.88/7) | 24.3% (1.70/7) | **+70.0% Detection Rate Increase** |

#### Terrain-Specific Performance (Hybrid VDN + Tabu):
*   **Easy Difficulty**:
    *   Coverage: **80.2%** (Hybrid VDN) vs 61.3% (Greedy)
    *   Latency: **4.2 steps** (Hybrid VDN) vs 31.8 steps (Greedy)
    *   Collisions: **7.0** (Hybrid VDN) vs 2284.0 (Greedy)
*   **Medium Difficulty**:
    *   Coverage: **80.7%** (Hybrid VDN) vs 63.5% (Greedy)
    *   Latency: **4.3 steps** (Hybrid VDN) vs 31.8 steps (Greedy)
    *   Collisions: **7.50** (Hybrid VDN) vs 1460.30 (Greedy)
*   **Hard Difficulty**:
    *   Coverage: **68.5%** (Hybrid VDN) vs 54.1% (Greedy)
    *   Latency: **53.7 steps** (Hybrid VDN) vs 55.3 steps (Greedy)
    *   Collisions: **2.30** (Hybrid VDN) vs 91.80 (Greedy)

---

### B. Raw VDN vs. Greedy Baseline
Without the Tabu Search memory extension, the raw VDN model maintains low collisions but suffers from trajectory lock-ups on complex grids:

*   **Average Coverage**: 41.3% (Raw VDN) vs 59.1% (Greedy)
*   **Average Latency**: 95.0 steps (Raw VDN) vs 81.9 steps (Greedy)
*   **Average Collisions**: **39.77** (Raw VDN) vs **1282.77** (Greedy)

> [!TIP]
> **Key Finding**: Integrating the Tabu search wrapper solves the local exploration lockup common in MARL grids, boosting coverage by **over 35%** while keeping collision counts near zero.

---

## 3. Directory Structure of the Backup

This folder (`VDN_Model_Backup`) contains the following consolidated assets:

```
VDN_Model_Backup/
│
├── latest.pt                      # Saved VDN Model Weights (Epoch 109 - optimized 11.4 MB)
│
├── PROJECT_SUMMARY.md             # This document (Architecture, metrics, guide)
│
├── mixed_eval_results.txt         # Raw metrics for VDN model vs Greedy baseline
├── mixed_eval_tabu_results.txt    # Raw metrics for Hybrid VDN + Tabu vs Greedy baseline
├── found_victims_gps.log          # GPS victim location alerts & rescue distances
├── venkateval.txt                 # Epoch-by-epoch training/curriculum evaluation logs
├── curriculum.log                 # Curriculum training phase step outputs
├── curriculum_proof.log           # Achievements proving graduation of curriculum difficulty phases
├── live_track.txt                 # Live exploration tracking coordinates and trajectory logs
│
├── airsim_wrapper.py              # Main 3D Simulator Interface, GPS, LiDAR, & Safety wrapper
├── test_takeoff.py                # Test script validating 4-drone spawning & mock fallback
├── run_airsim_exploration.bat      # One-click Windows startup script for the simulation runs
│
├── config/                        # Hyperparameters and environment configuration files
├── environment/                   # 2D grid/obstacle maps and coordinate system builders
└── baseline/                      # Greedy/Random baseline search policies
```

---

## 4. How to Deploy and Run

Follow these steps to restore and execute the simulation using the backed-up assets:

### Step A: Unreal Engine 5.7 Setup
1. Open the Epic Games Launcher and launch **Unreal Engine 5.7.4**.
2. Open your project (e.g., `finalyearvenkat`).
3. If prompted to rebuild the `AirSim` plugin, click **Yes** (ensure **.NET 8.0 Desktop Runtime** is installed on your Windows system).
4. In the UE Editor, go to **World Settings** (Window -> World Settings) and set `GameMode Override` to `AirSimGameMode`.
5. Ensure a `Player Start` actor is placed in the center of your environment at coordinates $(0,0,0)$.

### Step B: Launching the Python Wrapper
1. Ensure the AirSim configuration file is placed correctly in your Windows user directory:
   `C:\Users\<username>\Documents\AirSim\settings.json` (use the JSON configuration from `config/settings.json`).
2. Run the simulator in Unreal Engine by clicking **Play**.
3. In a terminal window within the project folder, run:
   ```bash
   python test_takeoff.py
   ```
   This will test connections to all 4 drones (`drone_0` to `drone_3`) and trigger takeoff and hover.
4. To execute the full cooperative exploration policy, double-click the **`run_airsim_exploration.bat`** file on your Desktop or run:
   ```bash
   python airsim_wrapper.py
   ```
5. Ground team rescue alerts and victim GPS coordinates will be written in real time to `found_victims_gps.log`.
