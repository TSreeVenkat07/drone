# PROJECT PROCEEDINGS: Phase 1 (2D Multi-UAV Search & Rescue)

## 1. Project Objective
To develop an intelligent, cooperative Multi-Agent Reinforcement Learning (MARL) framework for Unmanned Aerial Vehicles (UAVs) to autonomously navigate disaster zones, maximize map coverage, minimize collisions, and rapidly locate victims using real-world geographical data.

---

## 2. Core Model Architecture
The system integrates cooperative Multi-Agent Reinforcement Learning (MARL) with search heuristics to coordinate a swarm of UAVs.

### A. Hybrid Tabu Search VDN Algorithm
Our primary agent uses a value-decomposing deep reinforcement learning architecture combined with execution-time memory heuristics:
- **Value Decomposition Network (VDN):** Operates under the Centralized Training, Decentralized Execution (CTDE) paradigm. It factorizes the joint team action-value function $Q_{total}(\mathbf{s}, \mathbf{a})$ as a linear summation of individual utility functions $Q_i(o_i, a_i)$ for each UAV $i$, solving the multi-agent credit assignment problem:
  $$Q_{total}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^{N} Q_i(o_i, a_i)$$
- **Dueling DQN Architecture:** Individual agent networks utilize a Dueling structure, splitting the representation into a state-value stream $V(s)$ and an action-advantage stream $A(s, a)$. This speeds up learning by separating state-value updates from action-specific adjustments.
- **Double Q-Learning (DDQN):** Mitigates overestimation bias in Q-learning by decoupling action selection (online network) from action evaluation (target network).
- **Prioritized Experience Replay (PER):** Samples transition experiences from a global buffer proportional to their temporal-difference (TD) error magnitude, focusing model learning on challenging transitions.
- **Action Masking:** An environment-aware logic that filters out physically invalid moves (e.g., flying into known obstacles, colliding with boundaries) by setting their Q-values to $-\infty$ (or $-10^9$) before the argmax selection, guaranteeing safety.
- **Tabu Search Memory Buffer:** To prevent trajectory lock-ups (where UAVs circle in local loops or dead-ends), each agent maintains a short-term tabu list of its last 4 coordinates. During exploitation, the highest-ranking Q-value action that leads to a non-tabu cell is chosen.

### B. Greedy Frontier-Based Baseline Heuristic
For performance comparison, we benchmark against a localized greedy explorer:
- **Frontier-Based Navigation:** UAVs assess their immediate 8-connected local neighborhood.
- **State Scoring:** Cells with zero visitation count (unexplored frontiers) are prioritized (scored as distance 0).
- **Collision Avoidance & Obstacles:** Utilizes environmental masking to discard actions leading to obstacles or out-of-bounds positions.
- **Fallback Rule:** If all local cells are visited, it moves towards the cell with the minimum coverage count.
- **Limitations:** Purely local, lacks multi-agent coordination, collision avoidance, or long-term path planning, leading to high collision rates and repetitive paths.

---

## 3. Environment Design & Technology Stack

### A. Custom OpenAI Gym Environment
- **Gymnasium (v0.29.1):** Built a custom environment from scratch for multi-agent simulation.
- **State Space:** Local FOV (Field of View) for each drone, merged with a global shared coverage map and UAV status.
- **Action Space:** 9 discrete movements (N, NE, E, SE, S, SW, W, NW, and Hover).
- **Reward Function:** Multi-objective formulation using version `v3_full_spectrum`. Penalizes collisions and steps taken, rewards territory coverage, and heavily rewards victim detection.

### B. Core Technology Stack
- **Deep Learning Framework:** PyTorch (v2.1.0) with CUDA acceleration and Automatic Mixed Precision (AMP / FP16) training support.
- **Numerical & Data Processing:** NumPy (v1.24.3), Pandas (v2.0.3), SciPy (v1.11.2) for analytical computations and spatial metrics.
- **Image Processing:** Pillow (v10.0.0) for parsing real-world post-disaster satellite imagery.
- **Visualization & Logging:** Matplotlib (v3.7.2) and Seaborn (v0.12.2) for metric graphs, TensorBoard (v2.14.0) for training metrics, and Custom Logger for GPS logs.
- **Serialization & Configurations:** PyYAML (v6.0.1) for modular environment, agent, and training configuration parameters.

---

## 4. Evaluation Stages & Datasets

### Stage A: Procedural Generation (Standard Testing)
- **Method:** Randomized 2D grid maps generating synthetic obstacles.
- **Scenarios:** 
  - Building Collapse (dense structural walls)
  - Flood (water boundaries)
  - Wildfire (spreading danger zones)
- **Result:** Proved the baseline viability of the VDN+Tabu model, achieving up to 90% faster victim discovery than the baseline.

### Stage B: Real-World Topologies (OSM, NASA, UN-SPIDER)
- **Method:** Hardcoded real-world geographic constraints into the grid.
- **Datasets Used:**
  - **OpenStreetMap (OSM):** Dense city blocks for urban collapse scenarios.
  - **NASA FIRMS:** Wildfire boundary logic mimicking wind-driven fire spread.
  - **UN-SPIDER:** Flood inundation masks mimicking satellite radar data.
- **Result:** Proved the UAVs could navigate realistic, non-random chokepoints and complex urban layouts without getting trapped.

### Stage C: xBD Challenge Dataset (Kaggle xView2)
- **Method:** Evaluated the trained model against massive, real-world post-disaster satellite imagery.
- **Data Pipeline:**
  - Downloaded the 23 GB xView2 dataset.
  - Parsed precise WKT (Well-Known Text) polygons of real buildings.
  - Converted building damage scores (1=undamaged, 4=destroyed) into the grid environment (where Level 4 = victim spawn zones).
- **Key Feature:** Implemented **GPS Coordinate Mapping**, translating the agent's 2D grid movements back into real-world Latitude and Longitude values.
- **Result:** The model successfully navigated massive 1024x1024 satellite grids (e.g., the Palu Tsunami containing over 1,500 destroyed buildings), outperforming the baseline by 60% in coverage and reducing latency to first victim by 94.5%.

---

## 5. Final 2D Deliverables & Logs
The 2D phase successfully generated the following production-ready outputs:
1. **Trained Checkpoints:** Saved PyTorch weights for the VDN networks.
2. **UAV Path Logs:** CSV files tracking the exact step-by-step coordinates of every drone for ground-team visualization.
3. **GPS Victim Logs:** CSV files outputting the precise real-world Latitude/Longitude coordinates of discovered victims with 80%+ accuracy against ground truth annotations.
4. **Master Metrics:** Comprehensive markdown tables comparing all scenarios (Procedural, OSM, xBD).

---

