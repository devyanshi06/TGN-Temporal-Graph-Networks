# 🔬 Temporal Graph Networks (TGN) for Deep Learning on Dynamic Graphs

This repository contains a clean, modular, and interactive implementation of the state-of-the-art **Temporal Graph Network (TGN)** architecture framework, based on the research paper by Rossi et al. (Twitter Research). 

The framework is evaluated on continuous-time bipartite interaction streams using the official **Stanford JODIE Wikipedia dataset** and features a live visual dashboard built using **Streamlit**.

## 🚀 Key Performance Results
* **Dataset Scale:** 9,227 nodes and 157,474 chronological dynamic interaction events.
* **Peak Performance:** Achieved a **98.25% Validation Average Precision (AP)** by Epoch 9.
* **Optimization Success:** Smooth gradient backpropagation tracking with training loss minimizing steadily from `0.7059` down to `0.2996`.

## 🛠️ Project Architecture
The codebase is strictly modularized into three architectural tiers:
1. `model.py`: Implements persistent node-wise temporal registers (`TGNMemory`) coupled with a continuous `TimeEncoder` layer to track message intervals without staleness.
2. `train_eval.py`: Manages strict chronological dataset routing loops (70% Train / 15% Val) and prevents future-information leaks by executing the anti-leakage batch message store step (**Section 3.2** of the paper).
3. `app.py`: Houses the interactive 3-tab frontend deployment UI for real-time model execution, telemetry streaming, and hyperparameter tuning.

## 💻 Installation & Quickstart

### 1. Clone the Workspace
```bash
git clone [https://github.com/devyanshi06/TGN-Temporal-Graph-Networks.git](https://github.com/devyanshi06/TGN-Temporal-Graph-Networks.git)
cd TGN-Temporal-Graph-Networks
