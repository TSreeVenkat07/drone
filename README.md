# 🚁 2D Multi-UAV Search and Rescue (SAR) System

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

A simulation and evaluation framework for deploying multiple Unmanned Aerial Vehicles (UAVs) in a 2D disaster environment for search and rescue operations. This system utilizes advanced algorithms (like Reinforcement Learning / VDN) to coordinate drone fleets, optimizing area exploration and victim discovery while avoiding dynamic obstacles.

## ✨ Key Features

- **Multi-Agent Coordination:** Decentralized decision-making enabling drones to share local observations and coordinate their paths.
- **Dynamic Obstacle Avoidance:** Real-time adaptation to hazards like spreading fires, floods, and structural debris.
- **Memory & Mapping:** Intelligent memory systems to ensure UAVs don't repetitively scan the same areas, maximizing search efficiency.
- **Automated Metric Generation:** Built-in tools (`generate_pdf.py`) to automatically generate comprehensive evaluation reports and PDFs.
- **xBD Dataset Evaluation:** Integration with standard disaster datasets (`xbd_evaluation`) for realistic benchmarking.

## 📁 Project Structure

```text
.
├── uav_sar/                # Core simulation environment and Multi-UAV logic
├── xbd_evaluation/         # Evaluation scripts against the xBD disaster dataset
├── eval_references/        # Reference baselines and metric configurations
├── generate_pdf.py         # Script to compile simulation metrics into a PDF report
├── copy_script.py          # Utility script for data management
└── final2d_metrics.pdf     # Example output report
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Installation

1. Clone the repository (or navigate to your project directory):
   ```bash
   cd final
   ```

2. Install the required dependencies:
   ```bash
   # Assuming you have a requirements.txt file
   pip install -r requirements.txt
   ```

## 🕹️ Usage

### 1. Running the Simulation
*(Adjust commands based on your actual entry points)*
To start the multi-UAV search and rescue simulation:
```bash
python uav_sar/main.py
```

### 2. Evaluating Performance
To run the evaluation scripts on the xBD dataset:
```bash
python xbd_evaluation/evaluate.py
```

### 3. Generating Reports
After running your simulations, you can compile the metrics into a comprehensive PDF:
```bash
python generate_pdf.py
```
This will produce a file similar to `final2d_metrics.pdf` in your root directory.

## 🧠 How it Works

1. **Initialization:** The system loads a 2D grid map representing the disaster zone and deploys a fleet of UAVs.
2. **Observation:** Each drone scans its immediate FOV (Field of View) for open paths, obstacles, and victims.
3. **Communication & Planning:** Drones exchange information and utilize cooperative algorithms (like Value-Decomposition Networks) to decide the most optimal non-overlapping flight paths.
4. **Execution & Update:** Drones move to their target locations, and the global map is updated with discovered victims and safe zones.
5. **Completion:** The mission ends when the entire map is explored or all victims are located, resulting in a fully mapped safe route and victim coordinates.

## 📊 Metrics & Evaluation

The system tracks several key performance indicators (KPIs) to evaluate search efficiency:
- **Coverage Rate:** Percentage of the total map explored over time.
- **Victim Discovery Time:** Average time taken to locate trapped individuals.
- **Collision Rate:** Number of times drones encountered obstacles or other drones.
- **Redundancy:** Amount of time wasted revisiting already explored areas.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
