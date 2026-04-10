# Forgetting-Aware Neural Network Pruning (FANP)

![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch)
![W&B](https://img.shields.io/badge/Weights_&_Biases-Tracked-FFBE00?style=flat-square&logo=weightsandbiases)

> **Learning What NOT to Forget During Compression**

## Overview

Traditional pruning methods rely on weight magnitude or gradients, making estimations *before* pruning occurs. **FANP (Forgetting-Aware Neural Network Pruning)** closes this loop by introducing *forgetting dynamics* as a first-class pruning signal. A neuron's true importance is measured empirically based on how severely the model forgets its representations when that neuron is removed.

### Core Novelty

- **Dynamic Importance Signal**: Utilizes forgetting rate under removal instead of static L1 norms.
- **Empirical Fisher & Gradient Variance**: Incorporates the full empirical Fisher and high-variance gradients to identify fragile (and thus important) pathways.
- **Post-hoc Loss Spikes**: Centralizes the measurement of information loss during the compression phase.

---

## Project Structure

The main implementation resides in the `fanp/` directory:

- `configs/`: YAML experiment configuration files.
- `experiments/`: Main pruning pipeline and component ablation scripts.
- `models/`: ResNet definitions and test architectures.
- `pruning/`: Core forgetting-aware pruning engines, indicators, and metrics.
- `training/`: Baseline modeling and evaluation routines.

---

## Quick Start

### 1. Environment Setup

```powershell
# Establish environment & install dependencies
cd fanp
pip install -r requirements.txt
```

### 2. Reproducing Results

Execute the pipeline stages manually:

```powershell
$env:PYTHONIOENCODING="utf-8"

# 1. Train the baseline model architecture
python train_baseline.py --config configs/base.yaml

# 2. Execute main FANP structured pruning pipeline
python experiments/main_experiment.py

# 3. Run component ablation studies
python experiments/ablations/component_ablation.py

# 4. Sync offline W&B runs (If tracking is enabled)
wandb sync --entity <your-entity> --project fanp wandb/offline-run-*
```

---

## Status

This project is structurally complete. It delivers a fully tracked pipeline establishing the viability of forgetting dynamics, rather than strict magnitude alone, as an effective signal for localized and structured deep neural network pruning.
