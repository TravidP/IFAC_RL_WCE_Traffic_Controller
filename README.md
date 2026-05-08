
# Distributionally Robust Multi-Agent Reinforcement Learning for Intelligent Traffic Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Flow](https://img.shields.io/badge/Framework-Flow-blue.svg)](https://flow-project.github.io/)

This repository contains the official implementation of the paper:  
**"Distributionally Robust Multi-Agent Reinforcement Learning for Intelligent Traffic Control"** *Presented at the IFAC 2026 World Congress, Busan, South Korea.*

**Authors:** Shuwei Pei, Joran Borger, Arda Kosay, Muhammed O. Sayin, and Saeed Ahmed.  
**Affiliation:** Engineering and Technology Institute Groningen (ENTEG), University of Groningen.

---

## 📖 Project Overview

Learning-based traffic signal control (TSC) algorithms often fail when encountering atypical or adversarial traffic patterns because they are typically optimized for average performance. This project introduces a **Distributionally Robust MARL (DR-MARL)** framework designed to handle traffic demand uncertainty.

### Core Methodology:
1.  **Baseline MARL**: Training agents using Proximal Policy Optimization (PPO) to manage traffic lights in a multi-agent setting.
2.  **Contextual-Bandit Worst-Case Estimator (CB-WCE)**: A dedicated estimator that identifies the most challenging mixtures of origin-destination (OD) traffic demands.
3.  **Robust Fine-tuning**: Improving the baseline agents by training them against the dynamically identified worst-case scenarios, resulting in a controller that excels even under extreme conditions.

**Key Results:** Tested on a $3 \times 3$ Athens grid (calibrated with PNEUMA data) and a Sioux Falls network, DR-MARL achieved up to **51% shorter queues** and **38% higher speeds** compared to standard MARL under worst-case demand profiles.

---

## 🗂 Repository Structure

The project is built on the [Flow](https://github.com/flow-project/flow) framework, using **SUMO** for simulation and **Ray RLlib** for reinforcement learning.

```text
├── flow/                   # Core framework files (environments, networks, controllers)
│   ├── envs/multiagent/    # Custom traffic environments for DR-MARL and CB-WCE
│   └── networks/           # Network definitions (Athens Grid, Sioux Falls)
├── examples/               # Execution scripts
│   ├── train.py            # Main script for training RL policies
│   ├── eval_marl_vs_drmarl.py # Evaluation and comparison script
│   └── exp_configs/rl/multiagent/ # Configuration files for experiments
├── eval_results/           # Generated metrics, CSVs, and performance plots
├── environment.yml         # Conda environment definition
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

### 1. Prerequisites
- **Python 3.7+**
- **SUMO**: Follow the [SUMO installation guide](https://sumo.dlr.de/docs/Installing.html). Ensure `SUMO_HOME` is set in your environment variables.

### 2. Setup Environment
We recommend using Conda:
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate flow

# Install additional dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Phase 1: Train Baseline MARL
Train the initial traffic signal controller using standard demand distributions:
```bash
python examples/train.py --exp_config multiagent_traffic_light_grid
```

### Phase 2: Identify Worst-Case Scenarios (CB-WCE)
Train the Contextual-Bandit estimator to find adversarial demand mixtures:
```bash
python examples/train.py --exp_config worst_estimator_training
```

### Phase 3: Robust Fine-tuning (DR-MARL)
Fine-tune your baseline model using the worst-case distributions identified by the CB-WCE. Update your config to point to the saved baseline and estimator checkpoints, then run:
```bash
python examples/train.py --exp_config multiagent_traffic_light_grid_robust
```

### Phase 4: Evaluation
Compare the Baseline MARL and DR-MARL across different traffic demand groups:
```bash
python examples/eval_marl_vs_drmarl.py
```
Results (plots and raw CSVs) will be saved in the `eval_results/` directory.

---

## 📝 Citation

If you use this code or refer to our findings, please cite our work:

```bibtex
@inproceedings{pei2026drmarl,
  title={Distributionally Robust Multi-Agent Reinforcement Learning for Intelligent Traffic Control},
  author={Pei, Shuwei and Borger, Joran and Kosay, Arda and Sayin, Muhammed O. and Ahmed, Saeed},
  booktitle={IFAC World Congress},
  year={2026},
  address={Busan, South Korea}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
