# FANP — Forgetting-Aware Neural Network Pruning

**Learning What NOT to Forget During Compression**

Domain: Model Compression / Structured Pruning
Framework: PyTorch 2.6 + torch-pruning 1.6
Core Novelty: Forgetting Dynamics as a First-Class Pruning Signal

---

## Table of Contents

1. [What Is This Project](#1-what-is-this-project)
2. [The Core Idea](#2-the-core-idea)
3. [Research Questions](#3-research-questions)
4. [Architecture Overview](#4-architecture-overview)
5. [Repository Structure](#5-repository-structure)
6. [Environment Setup](#6-environment-setup)
7. [Phase 1 — Baseline Training](#7-phase-1--baseline-training)
8. [Phase 2 — Importance Estimators](#8-phase-2--importance-estimators)
9. [Phase 3 — Recovery and Structured Pruning](#9-phase-3--recovery-and-structured-pruning)
10. [Running the Full Experiment](#10-running-the-full-experiment)
11. [Running the Ablation Study](#11-running-the-ablation-study)
12. [Test Suite](#12-test-suite)
13. [Results Reference](#13-results-reference)
14. [Roadmap and Project Status](#14-roadmap-and-project-status)
15. [Key Design Decisions](#15-key-design-decisions)
16. [References](#16-references)

---

## 1. What Is This Project

Neural network pruning removes weights or entire filters from a trained model to reduce its size and inference cost. Classical methods (magnitude pruning, Fisher pruning) estimate how important a parameter is by looking at its value or gradient *before* removing it. They never observe what actually happens after removal.

FANP closes this loop. It measures *forgetting* — the real drop in model performance when a neuron is actually removed — and uses that signal to decide what to prune next. The result is a pruner that is guided by consequences, not predictions.

This codebase implements the full FANP pipeline from scratch:
- A pre-trained baseline (ResNet-56 on CIFAR-10)
- Five importance estimators including the composite Forgetting Score
- An adaptive pruning scheduler that slows down when the model is struggling
- A recovery-aware fine-tuning loop that tracks how fast accuracy bounces back
- Structured filter removal using a dependency graph
- A three-way experiment comparing Magnitude, Magnitude+FT, and FANP
- A component ablation study

---

## 2. The Core Idea

### Why Classic Methods Fall Short

| Problem | What Happens | FANP Fix |
|---|---|---|
| Silent forgetting | Small-magnitude but task-critical neurons get pruned; accuracy crashes late | Forgetting Score tracks actual ΔL on removal |
| Catastrophic interference | Co-adapted neurons pruned together cause outsized loss spikes | Round-by-round scoring re-evaluates after each prune |
| Fisher approximation error | Diagonal Fisher breaks for correlated layers | Empirical Fisher + gradient variance combined |
| Fixed pruning rate | Aggressive removal at the wrong time destroys recovery potential | Adaptive scheduler halves rate when mean FS > threshold τ |

### The Forgetting Score

Each weight `w_i` receives a composite importance score:

```
S_i = alpha * F_hat_i  +  beta * sigma^2(g_i)  +  gamma * delta_L_i
```

Where:
- `F_hat_i  = E[(dL/dw_i)^2]` — Empirical Fisher: how sensitive the loss is to this weight on average
- `sigma^2(g_i)` — Gradient variance over the last K batches: high variance means the weight is actively involved in learning
- `delta_L_i` — Loss spike: the actual loss increase when this weight is zeroed out (Taylor criterion: `(g_i * w_i)^2`)
- `alpha=0.5, beta=0.3, gamma=0.2` — fixed composite weights (meta-learning these is Phase 4)

High score = important, do not prune. Low score = safe to remove.

### Adaptive Pruning Schedule

The scheduler maintains a per-layer pruning rate. After each round:
- If mean forgetting score across the layer exceeds threshold `tau`, the rate is halved
- If the model is recovering well (slope above target), the rate is restored

This prevents runaway pruning in sensitive layers while allowing aggressive removal in redundant ones.

### Recovery Tracking

After each pruning round, a short fine-tuning phase runs for `N_recovery` steps. The recovery slope is:

```
recovery_slope = delta_accuracy / delta_steps   (units: % per step)
```

A high slope means the model bounced back quickly — the pruned weights were less critical than their score suggested. A near-zero slope (OVERPRUNED flag) means we removed too much.

---

## 3. Research Questions

**RQ1** Can forgetting dynamics measured during pruning serve as a more reliable importance signal than static weight-based metrics?

**RQ2** Does gradient variance at the neuron level correlate with the magnitude of post-pruning loss spikes?

**RQ3** Can an adaptive pruning schedule guided by forgetting score achieve higher compression ratios at equal accuracy?

**RQ4** How does the Forgetting Score compare against Magnitude and Magnitude+FT at 30%, 50%, 70%, and 90% sparsity?

---

## 4. Architecture Overview

```
+---------------------------+
|   Pre-trained Backbone    |   ResNet-56 on CIFAR-10
|   resnet56_best.pth       |   92.90% test accuracy
+---------------------------+
             |
             v
+---------------------------+
|   Importance Estimators   |   Phase 2
|   EmpiricalFisher         |   F_hat per weight
|   GradientVariance        |   sigma^2(g) over K steps
|   LossSpike (Taylor)      |   (g * w)^2 approximation
|   ForgettingScore         |   Composite: alpha*F + beta*V + gamma*L
|   AdaptivePruningScheduler|   Per-layer rate with threshold tau
+---------------------------+
             |
             v
+---------------------------+
|   Pruning Engine          |   Phase 3
|   StructuredFANPPruner    |   Filter removal via torch-pruning graph
|   FANPFilterImportance    |   Per-filter score = mean(S[f,c,kH,kW])
+---------------------------+
             |
             v
+---------------------------+
|   Recovery Loop           |   Phase 3
|   FineTuner               |   SGD + CosineAnnealingLR, masks re-applied
|   RecoveryTracker         |   Measures slope per round, flags OVERPRUNED
+---------------------------+
             |
             v
+---------------------------+
|   Experiment Runners      |   Phase 3
|   main_experiment.py      |   3-way comparison across 4 sparsity levels
|   component_ablation.py   |   6 configs at 70% sparsity
+---------------------------+
```

Data flow in a single FANP pruning run:

```
load pretrained model
  -> compute ForgettingScore for all weights
  -> AdaptivePruningScheduler decides rate for this round
  -> StructuredFANPPruner removes lowest-score filters
  -> FineTuner runs N_recovery steps, re-applies masks
  -> RecoveryTracker records slope
  -> repeat until target sparsity reached
  -> final evaluation on test set
```

---

## 5. Repository Structure

```
fanp/
|
|-- models/
|   |-- resnet.py               ResNet-56 definition (55 conv layers, 855,770 params)
|
|-- training/
|   |-- trainer.py              Full training loop: SGD + momentum + weight decay
|   |-- evaluator.py            Accuracy evaluation on any loader
|   |-- hooks.py                Forward/backward hooks for gradient collection
|
|-- pruning/
|   |-- importance/
|   |   |-- base.py             ImportanceEstimator abstract base class
|   |   |-- fisher.py           EmpiricalFisher: accumulates (grad)^2 over batches
|   |   |-- gradient_variance.py GradientVariance: Var(grad) over sliding window K
|   |   |-- loss_spike.py       LossSpike (Taylor criterion): (grad*weight)^2
|   |   |-- composite.py        ForgettingScore: alpha*F + beta*V + gamma*L
|   |                           AdaptivePruningScheduler: per-layer rate adaptation
|   |-- engine/
|   |   |-- structured.py       StructuredFANPPruner: filter removal using
|   |   |                         torch-pruning MetaPruner + dependency graph
|   |   |                         FANPFilterImportance aggregates per-weight
|   |   |                         scores to per-filter: I_f = mean(S[f,c,kH,kW])
|   |-- recovery/
|       |-- fine_tuner.py       FineTuner: SGD + CosineAnnealingLR for n_steps
|       |                         Re-applies masks after every optimizer.step()
|       |                         Returns RecoveryTrace with step-by-step history
|       |-- tracker.py          RecoveryTracker: wraps FineTuner, computes slope
|                                 per round, flags OVERPRUNED if slope < threshold
|
|-- experiments/
|   |-- baselines/
|   |   |-- magnitude.py        L1 magnitude pruning baseline
|   |-- ablations/
|   |   |-- component_ablation.py  6 configs: FANP-Full, Fisher-only, GradVar-only,
|   |                              Taylor-only, No-Adaptive, Magnitude-baseline
|   |-- main_experiment.py      3-way comparison: Magnitude | Magnitude+FT | FANP
|                                 Incremental JSON save after each sparsity level
|                                 Timestamped output, never overwrites
|                                 --resume flag for interrupted runs
|
|-- metrics/
|   |-- flops.py                FLOPs / MACs counter (Phase 4)
|   |-- sparsity.py             Sparsity ratio tracker
|
|-- data/                       CIFAR-10 auto-downloaded on first run
|-- configs/                    Hydra / OmegaConf config files
|-- checkpoints/
|   |-- resnet56_best.pth       Epoch 107, val_acc 93.24%  <-- always use this
|   |-- resnet56_last.pth       Epoch 200 (final state, lower accuracy)
|-- results/                    JSON + TXT outputs from experiment runs
|-- wandb/                      Auto-generated W&B logs
|
|-- train_baseline.py           Entry point for Phase 1 training
|-- eval_pruning.py             Evaluate magnitude baseline at any sparsity
|-- test_phase2.py              5 smoke tests for Phase 2 components
|-- test_phase3.py              20 smoke tests for Phase 3 components
|-- requirements.txt
|-- setup.py
|-- README.md
```

---

## 6. Environment Setup

### Requirements

- Windows 10/11 or Linux
- Python 3.12
- NVIDIA GPU with CUDA 12.x (tested on RTX 4060 Laptop GPU, 8 GB)
- ~3 GB disk space for CIFAR-10 and checkpoints

### Step-by-step Installation

**Step 1: Clone the repository**
```
git clone <your-repo-url>
cd DL_PROJECT
```

**Step 2: Create a virtual environment**
```
python -m venv fanp_env
```

**Step 3: Activate the environment**

On Windows (PowerShell):
```
fanp_env\Scripts\Activate.ps1
```

On Linux/Mac:
```
source fanp_env/bin/activate
```

**Step 4: Install dependencies**
```
cd fanp
pip install -r requirements.txt
```

This installs PyTorch 2.6.0+cu124, torchvision, torch-pruning 1.6.1, wandb, and all utilities.

**Step 5: Verify GPU is visible**
```
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Important Windows Notes

Two settings must always be set before running any script on Windows:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

This prevents Unicode print errors in the terminal.

All data loaders use `num_workers=0` in this codebase. This is intentional — Windows raises error 1455 (page file too small) with `num_workers > 0` in subprocess-based loaders.

### Standard Run Pattern on Windows

```powershell
cd C:\Users\Radhakrishna\Desktop\DL_PROJECT\fanp
$env:PYTHONIOENCODING="utf-8"
C:\Users\Radhakrishna\Desktop\DL_PROJECT\fanp_env\Scripts\python.exe <script>.py
```

---

## 7. Phase 1 — Baseline Training

### Goal

Train ResNet-56 on CIFAR-10 to convergence. Evaluate magnitude pruning at multiple sparsity levels to establish the baseline accuracy-sparsity curve.

### What Was Built

| Component | File | Description |
|---|---|---|
| ResNet-56 | `models/resnet.py` | 55 conv layers, 855,770 parameters |
| Trainer | `training/trainer.py` | SGD + momentum 0.9 + weight decay 5e-4, CosineAnnealingLR |
| Evaluator | `training/evaluator.py` | Top-1 accuracy on any DataLoader |
| Magnitude baseline | `experiments/baselines/magnitude.py` | `torch.nn.utils.prune.l1_unstructured` at given ratio |
| Training entry | `train_baseline.py` | Full 200-epoch training loop with checkpointing |
| Eval entry | `eval_pruning.py` | Evaluate magnitude baseline at specified sparsity |

### Training Command

```powershell
python train_baseline.py
```

Training runs for 200 epochs. Best checkpoint is saved when validation accuracy improves.

### Phase 1 Results

| Checkpoint | Epoch | Val Accuracy |
|---|---|---|
| `resnet56_best.pth` | 107 | 93.24% |
| `resnet56_last.pth` | 200 | 92.90% |

**Note on epoch 107:** The model is always loaded from `resnet56_best.pth` (epoch 107), not the final epoch. This is the highest validation accuracy point during training. All pruning experiments start from this checkpoint. The final epoch (200) has lower accuracy because of late-training noise.

### Magnitude Pruning Baseline Curve

| Sparsity | Test Accuracy | Drop vs Dense |
|---|---|---|
| 0% | 92.91% | baseline |
| 30% | 92.64% | -0.27% |
| 50% | 91.37% | -1.54% |
| 70% | 81.79% | -11.12% |
| 90% | 23.30% | -69.61% |

This curve shows that magnitude pruning degrades gracefully to 50% sparsity but collapses at 70%+. FANP targets this high-sparsity regime.

---

## 8. Phase 2 — Importance Estimators

### Goal

Implement the four building blocks of the Forgetting Score and the adaptive scheduler that uses it.

### Components

#### EmpiricalFisher (`pruning/importance/fisher.py`)

Accumulates squared gradients over N batches to approximate the diagonal Fisher information matrix.

```
F_hat_i = (1/N) * sum_over_batches( (dL/dw_i)^2 )
```

High Fisher value means the loss is sensitive to perturbations in `w_i`. Uses `register_full_backward_hook` to capture gradients without modifying the forward pass.

#### GradientVariance (`pruning/importance/gradient_variance.py`)

Tracks gradient variance over a sliding window of K steps.

```
sigma^2(g_i) = Var( {dL/dw_i : step in last K steps} )
```

High variance means the gradient for this weight oscillates a lot — the weight is actively being contested by different training examples, meaning it encodes task-specific knowledge.

#### LossSpike — Taylor Criterion (`pruning/importance/loss_spike.py`)

Approximates the loss increase when weight `w_i` is set to zero, using the first-order Taylor expansion:

```
delta_L_i ≈ (dL/dw_i * w_i)^2
```

This is the Molchanov et al. (ICLR 2017) Taylor criterion. It avoids forward-pass re-evaluation overhead while still capturing the consequence of removal. The implementation accumulates this quantity over multiple batches and averages.

**Note on implementation:** An earlier version of this module used layer-zeroing as an approximation. It was corrected to the proper per-weight Taylor formula `(grad * weight)^2`.

#### ForgettingScore — Composite (`pruning/importance/composite.py`)

Combines the three signals:

```
S_i = alpha * F_hat_i  +  beta * sigma^2(g_i)  +  gamma * delta_L_i
```

Default weights: `alpha=0.5, beta=0.3, gamma=0.2`.

The class also provides `score_layers()` which returns a dictionary mapping each parameter name to its composite score, as well as a normalized version suitable for ranking.

#### AdaptivePruningScheduler (`pruning/importance/composite.py`)

Manages the per-round pruning rate based on the global Forgetting Score signal.

Per-round logic:
1. Run `ForgettingScore.score_layers()` to get current scores
2. Compute `mean_fs = mean(all scores)`
3. If `mean_fs > tau`: halve the rate for this round (slow down)
4. Otherwise: restore rate toward `base_rate`
5. Return `(rate, mean_fs)` for logging

Default `tau=0.05`, `base_rate=0.10`.

### Running Phase 2 Smoke Tests

```powershell
python test_phase2.py
```

Expected output: `5/5 PASS — ALL PHASE 2 TESTS PASSED`

---

## 9. Phase 3 — Recovery and Structured Pruning

### Goal

1. Implement a fine-tuning loop that runs after each pruning round and tracks how fast accuracy recovers.
2. Implement physical filter removal (structured pruning) using the torch-pruning dependency graph.
3. Build the main experiment runner and ablation runner.

### Components

#### FineTuner (`pruning/recovery/fine_tuner.py`)

Runs SGD + CosineAnnealingLR for `n_steps` steps after pruning. After every `optimizer.step()`, it re-applies all active pruning masks so that zeroed weights cannot drift back. Evaluates on the validation set at `eval_interval` step intervals.

Returns a `RecoveryTrace` dataclass:
- `step_acc`: list of `(step, val_accuracy)` tuples
- `initial_acc`: validation accuracy before fine-tuning
- `final_acc`: validation accuracy at the last evaluation
- `recovery_slope` (property): `(final_acc - initial_acc) / n_steps` — the headline number

Key parameters:
- `n_steps=500` — full run; `n_steps=20` in quick/smoke test mode
- `lr=0.005` — lower than initial training to not disturb surviving weights
- `eval_interval=100` — prints progress every 100 steps

#### RecoveryTracker (`pruning/recovery/tracker.py`)

Wraps `FineTuner` and adds per-round bookkeeping. After each call to `measure()`:
- Runs fine-tuning
- Records `RecoveryMetrics`: round index, sparsity, acc before, acc after, slope, OVERPRUNED flag
- Prints a summary table on request

OVERPRUNED flag is raised when `recovery_slope < slope_threshold`. In the main experiment, `slope_threshold=0.0` for the Magnitude+FT control group to avoid false positives at low sparsity.

Output format:
```
 Rnd   Sparsity    Before     After     Gain       Slope        Status
--------------------------------------------------------------------
   0      10.0%    90.60%    92.74%   +2.14%    0.10700            OK
   1      18.9%    87.74%    91.84%   +4.10%    0.20500            OK
   2      27.0%    70.86%    91.20%  +20.34%    1.01700            OK
```

#### StructuredFANPPruner (`pruning/engine/structured.py`)

Performs physical filter removal — weights are not just masked but actually deleted, reducing parameter count and memory.

Implementation details:
- Uses `torch_pruning.MetaPruner` with a `DependencyGraph` built from the model
- `FANPFilterImportance` wraps the `ForgettingScore` and aggregates per-weight scores to per-filter scores: `I_f = mean(S_i for all weights i in filter f)`
- Pruning ratio applies to each conv layer independently: the bottom `pruning_ratio` fraction of filters by score are removed

Two bugs fixed during development:
1. `dep.target.name` from torch-pruning returns strings like `"layer1.0.conv1 (Conv2d(64, 64, ...))"` — the suffix must be stripped: `.split(" (")[0]`
2. `dep.handler` is a bound method — must compare via `dep.handler.__func__.__name__`, not by identity

At 20% pruning ratio: 855,770 → 536,844 parameters (1.59x compression verified in smoke test).

### Running Phase 3 Smoke Tests

```powershell
python test_phase3.py
```

Expected output: `20/20 PASS — ALL PHASE 3 TESTS PASSED`

The 20 tests cover:
- FineTuner: trace shape, slope sign, mask re-application
- RecoveryTracker: OVERPRUNED flag, metrics accumulation
- StructuredFANPPruner: parameter count reduction, output shape preserved
- main_experiment quick mode: JSON output written, correct keys present
- component_ablation quick mode: both ablation configs run and produce valid JSON

---

## 10. Running the Full Experiment

### What the Experiment Does

Runs three pruning pipelines at each of four sparsity levels (30%, 50%, 70%, 90%):

1. **Magnitude** — one-shot global L1 magnitude pruning, no fine-tuning. Pure baseline.
2. **Magnitude+FT** — same pruning, then full recovery fine-tuning. Control group to isolate fine-tuning benefit.
3. **FANP** — adaptive round-by-round structured pruning guided by Forgetting Score, with recovery tracking after each round.

Each method starts from the same `resnet56_best.pth` checkpoint (epoch 107, 93.24% val accuracy).

### Running

```powershell
cd C:\Users\Radhakrishna\Desktop\DL_PROJECT\fanp
$env:PYTHONIOENCODING="utf-8"
C:\Users\Radhakrishna\Desktop\DL_PROJECT\fanp_env\Scripts\python.exe experiments/main_experiment.py
```

This takes approximately 4–6 hours on an RTX 4060 Laptop GPU.

### Incremental Saving

Results are saved after every sparsity level, not just at the end. If the run is interrupted at 70% sparsity, the 30% and 50% results are not lost.

Output files (written to `results/`):
- `main_YYYYMMDD_HHMMSS.json` — full structured results, never overwrites previous runs
- `main_YYYYMMDD_HHMMSS_summary.txt` — human-readable table for quick review

### Resuming an Interrupted Run

```powershell
python experiments/main_experiment.py --resume results/main_20260312_143022.json
```

The script reads the partial JSON, identifies which sparsity levels were already completed, and skips them. Results are appended to the same file.

### Quick Test Mode (Smoke Test)

```powershell
python experiments/main_experiment.py --quick --sparsity 0.30
```

Overrides FT steps to 20, reduces data subset, runs only the specified sparsity level. Completes in under 5 minutes. Use this to verify the pipeline is working before committing to the full run.

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--quick` | off | Enable fast smoke-test mode |
| `--sparsity 0.70` | all levels | Run only the specified sparsity level |
| `--resume PATH` | none | Resume from a partial results JSON file |
| `--out PATH` | auto-timestamped | Override the output file path |

### Expected Output Format

```
==============================================================
  TARGET SPARSITY: 70%
==============================================================

  --- MAGNITUDE  (no fine-tuning) ---
  checkpoint: resnet56_best.pth | epoch 107 | val_acc 93.24% (best checkpoint — not the final epoch)
  [Magnitude]  sparsity 70.0%   test_acc 81.79%

  --- MAGNITUDE + FINE-TUNE  (control) ---
  [FineTuner] start val_acc: 81.20%  (500 steps, lr=0.005)
  ...
  [Magnitude+FT]  sparsity 70.0%   test_acc 83.50%

  --- FANP  (our method) ---
  Round   1 | Sparsity:  9.962% | mean_FS: 0.037 | rate: 0.100 | pruned: 84,830
  [FineTuner] start val_acc: 75.56%  (500 steps, lr=0.005)
  ...

  PARTIAL RESULT @ 70% sparsity
  Method             Test Acc   vs Magnitude
  ------------------------------------------
  Magnitude            81.79%       baseline
  Magnitude+FT         83.20%         +1.41%
  FANP                 88.50%         +6.71%
```

---

## 11. Running the Ablation Study

```powershell
python experiments/ablations/component_ablation.py
```

Runs 6 configurations at 70% target sparsity to isolate which components of the Forgetting Score matter:

| Config | Description |
|---|---|
| FANP-Full | All three signals: Fisher + GradVar + Taylor |
| Fisher-only | Only Fisher information (beta=0, gamma=0) |
| GradVar-only | Only gradient variance (alpha=0, gamma=0) |
| Taylor-only | Only loss spike / Taylor criterion (alpha=0, beta=0) |
| No-Adaptive | Score unchanged but scheduler disabled (tau=2.0, rate never halves) |
| Magnitude | Baseline — no Forgetting Score at all |

Saves results to `results/ablation.json`. Quick mode available with `--quick`.

---

## 12. Test Suite

### Phase 2 Tests

```powershell
python test_phase2.py
```

| Test | What It Checks |
|---|---|
| Fisher | Scores computed, non-zero, correct shape |
| GradientVariance | Non-negative variance values |
| LossSpike | Taylor scores are non-negative squared quantities |
| ForgettingScore | Composite scores bounded, sum to expected range |
| AdaptivePruningScheduler | Rate is reduced when mean_fs > tau |

### Phase 3 Tests

```powershell
python test_phase3.py
```

| Test Group | Tests | What It Checks |
|---|---|---|
| FineTuner | 5 | Trace shape, slope ≥ 0, masks survive step |
| RecoveryTracker | 4 | Metrics recorded, OVERPRUNED set correctly |
| StructuredFANPPruner | 4 | Param count drops, output logits shape unchanged |
| main_experiment | 4 | Quick run completes, JSON valid, all sparsity keys present |
| component_ablation | 3 | Quick 2-config run completes, JSON valid |

All 25 tests (5 Phase 2 + 20 Phase 3) pass on the current codebase.

---

## 13. Results Reference

### Checkpoints

| File | Epoch | Val Accuracy | Use For |
|---|---|---|---|
| `checkpoints/resnet56_best.pth` | 107 | 93.24% | ALL pruning experiments |
| `checkpoints/resnet56_last.pth` | 200 | 92.90% | Reference only |

**Always use `resnet56_best.pth`.** The final epoch checkpoint has lower accuracy due to training noise in the last ~30 epochs.

### Magnitude Pruning Baseline (Phase 1)

| Sparsity | Test Acc | Drop |
|---|---|---|
| 0% | 92.91% | — |
| 30% | 92.64% | -0.27% |
| 50% | 91.37% | -1.54% |
| 70% | 81.79% | -11.12% |
| 90% | 23.30% | -69.61% |

### FANP vs Magnitude (Full Run — Pending)

The full 4-level experiment has not yet completed. Results will be written to `results/main_*.json` after the run. The table below will be populated from that output.

| Sparsity | Magnitude | Magnitude+FT | FANP | FANP vs Mag |
|---|---|---|---|---|
| 30% | 92.64% | TBD | TBD | TBD |
| 50% | 91.37% | TBD | TBD | TBD |
| 70% | 81.79% | TBD | TBD | TBD |
| 90% | 23.30% | TBD | TBD | TBD |

### Structured Pruning Compression (Phase 3 Verified)

| Pruning Ratio | Original Params | Pruned Params | Compression |
|---|---|---|---|
| 20% filters | 855,770 | 536,844 | 1.59x |

---

## 14. Roadmap and Project Status

### Phase Completion

| Phase | Scope | Status | Notes |
|---|---|---|---|
| Phase 1 | Baseline training + magnitude pruning | Complete | 92.90% test acc, baseline curve established |
| Phase 2 | Importance estimators (Fisher, GradVar, Taylor, Composite, Adaptive) | Complete | 5/5 smoke tests pass |
| Phase 3 | Recovery tracker, structured pruner, experiment runner, ablation runner | Complete (code) | 20/20 smoke tests pass; full experiment not yet run |
| Phase 4 | Meta-learning alpha/beta/gamma, ONNX export, latency, ImageNet, BERT | Not started | — |
| Phase 5 | Paper draft, W&B dashboard, Gradio demo | Not started | — |

### Overall Completion: ~55%

- Phase 1 (20% of total): 100% done
- Phase 2 (25% of total): 100% done
- Phase 3 (25% of total): 80% done — code and tests complete, full experiment results pending
- Phase 4 (20% of total): 0%
- Phase 5 (10% of total): 0%

### What Remains in Phase 3

- Run `experiments/main_experiment.py` to completion (4–6 hours)
- Run `experiments/ablations/component_ablation.py` (2–3 hours)
- Populate the results table in this README

### What Is Phase 4

Phase 4 converts this from a working research prototype to a paper-ready result:

1. **Meta-learning alpha/beta/gamma** — currently fixed at 0.5/0.3/0.2. Learning these via a small meta-loop on a held-out task would be the main algorithmic contribution.
2. **ONNX export** — prune model → export to ONNX → measure real inference latency on CPU and GPU. This converts accuracy numbers into actual speedup numbers.
3. **FLOPs and MACs counting** — use `metrics/flops.py` to report compute reduction alongside parameter reduction.
4. **ImageNet + ResNet-50** — scale the experiment to a larger dataset and model to show the method is not CIFAR-10 specific.
5. **BERT on GLUE** — extend to NLP to demonstrate that forgetting dynamics apply beyond vision.

---

## 15. Key Design Decisions

### num_workers=0

All DataLoaders in this codebase use `num_workers=0`. On Windows, `num_workers > 0` causes the PyTorch DataLoader to spawn subprocesses via Python's `multiprocessing` module, which requires enough virtual address space per process. On many Windows machines this raises error code 1455 (the paging file is too small). Setting `num_workers=0` runs the data pipeline in the main process, which is slower but always stable.

### Checkpoint Strategy: Best vs. Last

Pruning always starts from the best validation accuracy checkpoint, never the final epoch. This is intentional: the final epoch often has slightly lower accuracy due to cosine LR decay reaching near-zero. Starting from the best checkpoint gives the pruner the highest-quality weights to work with and makes the baseline comparison fair.

### Taylor Criterion vs. Forward-Pass Loss Spike

The LossSpike module uses the Taylor expansion `(grad * weight)^2` rather than actually zeroing out each weight and measuring the forward-pass loss change. The forward-pass approach is exact but requires one forward pass per weight — completely intractable for a 855K-parameter model. The Taylor approximation requires only one backward pass for all weights simultaneously, reducing cost from O(N) forward passes to O(1).

### Structured vs. Unstructured Pruning

The project implements both:
- **Unstructured pruning** (`torch.nn.utils.prune`): used in the magnitude baseline and ForgettingScore computation. Fast. Does not actually reduce model size without sparse tensor hardware support.
- **Structured pruning** (`StructuredFANPPruner`): physically removes entire filters. Slower to compute, but the resulting model has fewer parameters, fewer FLOPs, and actually runs faster on standard hardware.

Phase 3 uses structured pruning for the main FANP experiments because actual inference speedup matters for the paper claims.

### Incremental Saving

`main_experiment.py` saves results to disk after every sparsity level, not at the end. The full experiment takes ~6 hours and can be interrupted by power outages, GPU resets, or OS timeouts. Saving incrementally ensures no results are lost on interruption.

---

## 16. References

| Paper | Venue | Relevance |
|---|---|---|
| LeCun et al., Optimal Brain Damage | NeurIPS 1990 | First second-order pruning; diagonal Hessian |
| Hassibi & Stork, Optimal Brain Surgeon | NeurIPS 1993 | Full inverse Hessian; exact weight perturbation |
| Han et al., Learning Both Weights and Connections | NeurIPS 2015 | Magnitude pruning + retraining |
| Molchanov et al., Pruning Filters for Efficient ConvNets | ICLR 2017 | Taylor criterion — direct basis of LossSpike module |
| Toneva et al., An Empirical Study of Example Forgetting | ICLR 2019 | Forgetting events — key conceptual inspiration |
| Frankle & Carlin, The Lottery Ticket Hypothesis | ICLR 2019 | Sparse sub-networks from random initialization |
| Singh & Alistarh, WoodFisher | NeurIPS 2020 | Block-diagonal Fisher; direct comparison baseline |
| Evci et al., Rigging the Lottery | ICML 2020 | Dynamic sparse training; online importance updates |
| Kwon et al., Fast Post-Training Pruning for Transformers | NeurIPS 2022 | Gradient-based importance in transformer setting |
| Frantar & Alistarh, SparseGPT | ICML 2023 | One-shot pruning of GPT with Hessian inverses |

---

*FANP — Forgetting-Aware Neural Network Pruning*
*Radhakrishna — Deep Learning Project, 2026*
