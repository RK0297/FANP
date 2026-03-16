*Forgetting-Aware Neural Network Pruning  |  Deep Learning Project*

**DEEP LEARNING PROJECT**

**Forgetting-Aware Neural Network Pruning**

***Learning What NOT to Forget During Compression***

|<p>**Domain**</p><p>Model Compression / Structured Pruning</p>|<p>**Level**</p><p>Basic → Advanced (Full Stack)</p>|
| :-: | :-: |
|<p>**Framework**</p><p>PyTorch + W&B + ONNX</p>|<p>**Core Novelty**</p><p>Forgetting Dynamics as Pruning Signal</p>|


# **1.  Novelty & Research Contribution**

|<p>**★  Core Insight**</p><p>Classic pruning removes weights by magnitude or gradient — treating all removals equally. This project introduces forgetting dynamics as a first-class pruning signal: a neuron's importance is measured by how badly the model 'forgets' when that neuron is removed.</p>|
| :- |

## **1.1  What Makes This Novel**
Existing pruning methods suffer from a fundamental mismatch: they estimate importance before pruning without observing post-pruning consequences. This work closes that loop.

|**Dimension**|**Prior Work**|**This Work (FANP)**|
| :- | :- | :- |
|**Importance Signal**|Weight magnitude / L1 norm|Forgetting rate under removal|
|**Measurement Timing**|Static — before pruning|Dynamic — during/after pruning|
|**Fisher Information**|Diagonal approximation only|Full empirical Fisher + gradient variance|
|**Gradient Variance**|Not used|High-variance = unstable = important|
|**Loss Spikes**|Ignored post-hoc|Central metric for forgetting score|
|**Recoverability**|Not modeled|Neurons scored on recovery trajectory|

## **1.2  Three Key Technical Contributions**
- Forgetting Score (FS): A composite metric combining loss spike magnitude, gradient variance, and Fisher information into a single neuron-level importance score.
- Adaptive Pruning Schedule: Pruning rate is dynamically adjusted based on the running forgetting score — slow down when forgetting is high, accelerate when stable.
- Recovery-Aware Importance: Instead of one-shot evaluation, the system tracks how quickly performance recovers post-pruning to re-score neurons.


# **2.  Problem Statement**

## **2.1  Formal Problem Definition**
Given a trained neural network f(x; θ) with parameters θ ∈ R^n and a dataset D = {(xᵢ, yᵢ)}, the goal of pruning is to find a sparse sub-network f(x; θ_mask) such that:

**argmin ||θ_mask||₀   subject to   L(f(x; θ_mask), y) ≤ L(f(x; θ), y) + ε**

where ||θ_mask||₀ is the number of non-zero parameters and ε is an acceptable performance degradation budget.

## **2.2  The Forgetting Problem**
Classical pruning methods estimate parameter importance independently of actual removal consequences. This leads to three failure modes:

1. Silent Forgetting: Neurons with small magnitude but high task relevance are pruned, causing gradual accuracy collapse that is only detected late.
1. Catastrophic Interference: Removing co-adapted neuron groups causes disproportionate loss spikes not predicted by individual scores.
1. Fisher Approximation Error: Diagonal Fisher assumptions break down for correlated neurons — common in attention heads and batch-norm layers.

## **2.3  Research Questions**
- RQ1: Can forgetting dynamics measured during pruning serve as a more reliable importance signal than static weight-based metrics?
- RQ2: Does gradient variance at the neuron level correlate with the magnitude of post-pruning loss spikes?
- RQ3: Can an adaptive pruning schedule guided by forgetting score achieve higher compression ratios at equal accuracy?
- RQ4: How does the Forgetting Score compare against OBD, OBS, and magnitude pruning on standard benchmarks?


# **3.  Literature Survey & Citation Map**

## **3.1  Foundational Pruning Methods**

|**Paper**|**Venue / Year**|**Key Contribution**|
| :- | :- | :- |
|*LeCun et al., Optimal Brain Damage*|NeurIPS 1990|First second-order pruning; diagonal Hessian for saliency|
|*Hassibi & Stork, Optimal Brain Surgeon*|NeurIPS 1993|Full inverse Hessian; exact weight perturbation|
|*Han et al., Learning Both Weights and Connections*|NeurIPS 2015|Magnitude pruning + retraining pipeline|
|*Frankle & Carlin, Lottery Ticket Hypothesis*|ICLR 2019|Sparse winning sub-networks from random init|
|*Frankle et al., Linear Mode Connectivity and LTH*|ICML 2020|Stability of winning tickets across SGD noise|

## **3.2  Fisher Information & Second-Order Methods**

|**Paper**|**Venue / Year**|**Relevance to This Work**|
| :- | :- | :- |
|*Molchanov et al., Pruning Filters for Efficient ConvNets*|ICLR 2017|Taylor expansion criterion — 1st-order Fisher approx|
|*Theis et al., Faster gaze prediction with dense networks and Fisher pruning*|arXiv 2018|Empirical Fisher for structured pruning|
|*Singh & Alistarh, WoodFisher: Efficient second-order approx for NNs*|NeurIPS 2020|Block-diagonal Fisher; direct baseline for this work|
|*Frantar & Alistarh, SPDY: Accurate Pruning with Scalable Second-Order*|MLSys 2022|SOTA second-order structured pruning|
|*Frantar et al., SparseGPT: Massive Language Models can be Pruned*|ICML 2023|One-shot pruning of GPT models with Hessian inverses|

## **3.3  Dynamic & Forgetting-Aware Methods**

|**Paper**|**Venue / Year**|**Connection to This Work**|
| :- | :- | :- |
|*Toneva et al., An Empirical Study of Example Forgetting*|ICLR 2019|Forgetting events in training — key conceptual inspiration|
|*You et al., Drawing Early-Bird Tickets*|ICLR 2020|Dynamic pruning at early training stages|
|*Evci et al., Rigging the Lottery: Making All Tickets Winners*|ICML 2020|Dynamic sparse training; online importance updates|
|*Mallya & Lazebnik, PackNet: Adding Multiple Tasks to a Neural Network*|CVPR 2018|Forgetting in continual learning — related metric design|

## **3.4  Gradient Variance & Training Dynamics**
- Hoffer et al., Train Longer, Generalize Better (NeurIPS 2017) — gradient noise scale as a signal of model capacity.
- Smith & Le, A Bayesian Perspective on Generalization (ICLR 2018) — learning rate and batch size effects on loss landscape curvature.
- Frankle et al., The Early Phase of Neural Network Training (ICLR 2020) — critical learning phases where pruning is most damaging.
- Kwon et al., A Fast Post-Training Pruning Framework for Transformers (NeurIPS 2022) — gradient-based importance for transformer pruning.


# **4.  System Architecture**

## **4.1  High-Level Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│          FANP — Forgetting-Aware Neural Network Pruning      │
└─────────────────────────────────────────────────────────────┘
                         │
  ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────┐
  │  Baseline Model  │   │  Importance     │   │  Pruning Engine  │
  │  (Pre-trained)   │──>│  Estimator      │──>│  (FANP Core)     │
  │  ResNet / VGG /  │   │  Fisher + Grad  │   │  Forgetting Score│
  │  BERT / ViT      │   │  Variance + FS  │   │  + Adaptive Rate │
  └──────────────────┘   └─────────────────┘   └──────────────────┘
                                                        │
  ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────┐
  │  Fine-tune /     │   │  Evaluation &   │   │  Recovery        │
  │  Recovery Loop   │<──│  Metrics Module │<──│  Tracker         │
  │  (Fine-grained)  │   │  Acc/Latency/   │   │  (re-scores      │
  │                  │   │  FLOPs/MACs     │   │   importance)    │
  └──────────────────┘   └─────────────────┘   └──────────────────┘
```

## **4.2  Core Components**
### **4.2.1  Importance Estimator**
The Importance Estimator computes a composite Forgetting Score (FS) for each neuron/filter using three signals:

|**Signal**|**Formula**|**Interpretation**|
| :- | :- | :- |
|**Gradient Variance**|σ²(∂L/∂wᵢ) over last K steps|High variance → neuron is actively used for task|
|**Fisher Information**|F̂ᵢ = E[(∂L/∂wᵢ)²]|Expected squared gradient = parameter sensitivity|
|**Loss Spike (FS)**|ΔL = L(f_pruned) - L(f_full)|Direct measurement of forgetting on removal|
|**Composite Score**|S = α·F̂ + β·σ²(g) + γ·ΔL|Learnable weights α, β, γ optimized via meta-learning|

### **4.2.2  Pruning Engine**
The Pruning Engine takes importance scores and applies structured or unstructured pruning with an adaptive rate:

- Structured Pruning: Entire filters/heads are removed based on composite score ranking.
- Unstructured Pruning: Individual weight masking for maximum sparsity, followed by sparse tensor inference.
- Adaptive Rate: If mean FS across a layer exceeds a threshold τ, pruning rate for that layer is halved for the next step.

### **4.2.3  Recovery Tracker**
After each pruning step, a lightweight fine-tuning phase runs for N_recovery steps. The Recovery Tracker measures:

- Recovery Slope: Δaccuracy / Δsteps during fine-tuning — fast recovery means the pruned neuron was less critical.
- Plateau Detection: If accuracy plateaus before reaching target, a re-scoring pass re-activates low-importance neurons.


# **5.  Technology Stack**

|**Category**|**Tool / Library**|**Purpose**|
| :- | :- | :- |
|**Core DL**|PyTorch 2.x|Model training, gradient hooks, masking|
||torchvision / HuggingFace|Pretrained backbones (ResNet, ViT, BERT)|
|**Compression**|torch.nn.utils.prune|Built-in unstructured pruning masks|
||torch-pruning (VainF)|Dependency graph for structured pruning|
|**Experiment Tracking**|Weights & Biases (W&B)|Real-time forgetting score visualization|
||TensorBoard|Gradient variance plots, loss spikes|
|**Profiling**|ONNX + ONNX Runtime|Export and benchmark pruned model latency|
||torch.profiler|FLOPs / MACs counting|
|**Datasets**|CIFAR-10/100, ImageNet, GLUE|Classification and NLP benchmarks|
|**Config Management**|Hydra + OmegaConf|Sweep configs for α, β, γ, τ hyperparameters|
|**Compute**|NVIDIA GPU (CUDA 12.x)|Training + profiling|
||Google Colab / Lambda Labs|Cloud GPU for large model experiments|
|**Testing**|pytest + pytest-cov|Unit tests for scoring functions|
|**Documentation**|Sphinx + GitHub Pages|Auto-generated API docs|


# **6.  Project Directory Structure**

Full project layout for the FANP codebase — organized for reproducibility and modularity:

```
fanp/                              # Root
├── configs/                        # Hydra config files
│   ├── base.yaml                   # Global hyperparams
│   ├── pruning/
│   │   ├── structured.yaml
│   │   └── unstructured.yaml
│   └── experiment/
│       ├── cifar10_resnet.yaml
│       └── imagenet_vit.yaml
├── data/
│   ├── __init__.py
│   ├── cifar.py                    # CIFAR-10/100 loaders
│   ├── imagenet.py                 # ImageNet loader + aug
│   └── glue.py                     # NLP benchmarks
├── models/
│   ├── __init__.py
│   ├── resnet.py                   # Custom ResNet with hooks
│   ├── vit.py                      # ViT with attention pruning
│   └── bert.py                     # BERT head pruning
├── pruning/
│   ├── __init__.py
│   ├── importance/
│   │   ├── base.py                 # ImportanceEstimator ABC
│   │   ├── fisher.py               # Empirical Fisher computation
│   │   ├── gradient_variance.py    # Grad variance tracker
│   │   ├── loss_spike.py           # ΔL forgetting measurement
│   │   └── composite.py            # FANP Forgetting Score
│   ├── engine/
│   │   ├── structured.py           # Filter/head removal
│   │   ├── unstructured.py         # Weight masking
│   │   └── adaptive_scheduler.py   # Rate-adaptive pruner
│   └── recovery/
│       ├── tracker.py              # Recovery slope measurement
│       └── fine_tuner.py           # Post-pruning fine-tune loop
├── training/
│   ├── trainer.py                  # Main training loop
│   ├── evaluator.py                # Acc / loss / latency eval
│   └── hooks.py                    # Forward/backward hooks
├── metrics/
│   ├── flops.py                    # FLOPs / MACs counter
│   ├── sparsity.py                 # Sparsity ratio tracker
│   └── forgetting.py               # FS logging utilities
├── experiments/
│   ├── baselines/
│   │   ├── magnitude.py            # L1 magnitude pruning
│   │   ├── random.py               # Random pruning baseline
│   │   └── woodfisher.py           # WoodFisher comparison
│   ├── ablations/
│   │   ├── no_fisher.py
│   │   ├── no_grad_var.py
│   │   └── no_recovery.py
│   └── main_experiment.py          # Full FANP pipeline
├── scripts/
│   ├── train_baseline.sh
│   ├── run_pruning.sh
│   └── export_onnx.sh
├── tests/
│   ├── test_fisher.py
│   ├── test_forgetting_score.py
│   ├── test_pruning_engine.py
│   └── test_recovery.py
├── notebooks/
│   ├── 01_explore_forgetting.ipynb
│   ├── 02_importance_vis.ipynb
│   └── 03_results_analysis.ipynb
├── requirements.txt
├── setup.py
├── README.md
└── wandb/                          # Auto-generated W&B logs
```


# **7.  Implementation Roadmap**

## **Phase 1 — Baseline (Week 1–2)**
1. Set up repository, install PyTorch + torch-pruning + W&B.
1. Train baseline ResNet-56 on CIFAR-10 to convergence (~93% top-1).
1. Implement and test magnitude pruning baseline; log sparsity vs accuracy curve.
1. Implement gradient hooks to capture per-neuron gradients during training.

## **Phase 2 — Core FANP (Week 3–5)**
1. Implement Empirical Fisher estimator (fisher.py) with diagonal and block-diagonal modes.
1. Implement Gradient Variance tracker over sliding window K.
1. Implement Loss Spike module: remove neuron → eval ΔL → restore neuron.
1. Compose Forgetting Score S = α·F̂ + β·σ²(g) + γ·ΔL with fixed then learned weights.
1. Build Adaptive Pruning Scheduler with per-layer rate adaptation.

## **Phase 3 — Recovery & Advanced Features (Week 6–8)**
1. Implement Recovery Tracker: measure recovery slope over N_recovery steps.
1. Add re-scoring pass: neurons with fast recovery are re-flagged for pruning.
1. Extend to structured pruning (filter removal) using torch-pruning dependency graph.
1. Scale to ImageNet + ResNet-50; implement ViT attention head pruning.

## **Phase 4 — Experiments & Evaluation (Week 9–11)**
1. Run full ablation: FANP vs. no-Fisher, no-grad-variance, no-recovery-tracking.
1. Compare against WoodFisher, SparseGPT baselines at 50%, 70%, 90% sparsity.
1. Export pruned models to ONNX; measure real-world inference latency.
1. Run NLP experiments on BERT + GLUE for broader applicability.

## **Phase 5 — Report & Demo (Week 12)**
1. Write research paper draft: problem, method, experiments, conclusions.
1. Create interactive W&B dashboard showing forgetting score evolution.
1. Build Gradio demo: upload any model → see pruning report + forgetting score map.


# **8.  Key Algorithm Pseudocode**

## **8.1  Forgetting Score Computation**

```python
def compute_forgetting_score(model, loader, neuron_i, alpha, beta, gamma, K=50):
    # 1. Collect gradients over K batches
    grads = collect_gradients(model, loader, neuron_i, steps=K)

    # 2. Fisher Information (empirical)
    F_hat = torch.mean(grads ** 2)          # E[(dL/dw)^2]

    # 3. Gradient Variance
    sigma_sq = torch.var(grads)             # Var(dL/dw)

    # 4. Loss Spike — temporarily remove neuron
    with prune_temporarily(model, neuron_i):
        delta_L = eval_loss(model, loader) - baseline_loss

    # 5. Composite Forgetting Score
    FS = alpha * F_hat + beta * sigma_sq + gamma * delta_L
    return FS
```
| :- |

## **8.2  Adaptive Pruning Loop**

```python
def fanp_prune(model, target_sparsity, tau=0.05):
    current_sparsity = 0.0
    while current_sparsity < target_sparsity:
        scores = {}
        for layer in model.prunable_layers():
            for neuron in layer.neurons():
                scores[neuron] = compute_forgetting_score(model, ...)

        # Adaptive rate: halve if mean FS is high (model is forgetting fast)
        mean_FS = torch.mean(torch.tensor(list(scores.values())))
        rate = base_rate if mean_FS < tau else base_rate / 2

        # Prune bottom-scoring neurons
        candidates = sorted(scores, key=scores.get)[:int(rate * n_neurons)]
        prune_neurons(model, candidates)

        # Recovery phase
        recovery_slope = fine_tune_and_track(model, steps=N_recovery)
        rescore_fast_recovery(model, recovery_slope, scores)

        current_sparsity = compute_sparsity(model)
    return model
```
| :- |


# **9.  Evaluation Plan & Expected Results**

## **9.1  Metrics**

|**Metric**|**Definition**|**Target**|
| :- | :- | :- |
|**Top-1 Accuracy Drop**|Acc_baseline - Acc_pruned|< 1% at 50% sparsity|
|**FLOPs Reduction**|(FLOPs_full - FLOPs_pruned) / FLOPs_full|> 60% at 70% sparsity|
|**Inference Speedup**|Latency_full / Latency_pruned (ONNX)|> 2x at 70% sparsity|
|**Recovery Steps**|Steps to reach baseline - 0.5%|< 500 steps per pruning iteration|
|**Forgetting Score Correlation**|Pearson r(FS, actual ΔL)|> 0.85 correlation|

## **9.2  Baselines to Beat**
- Magnitude Pruning (Han et al., 2015) — most widely used industry baseline.
- WoodFisher (Singh & Alistarh, 2020) — current SOTA second-order structured pruning.
- Random Pruning — lower bound on performance.
- One-Shot SparseGPT — for transformer architecture experiments.

## **9.3  Ablation Study Design**

|**Ablation**|**Component Removed**|**Expected Effect**|
| :- | :- | :- |
|FANP - Fisher|Fisher Information term (α=0)|Higher acc. drop at structured pruning|
|FANP - GradVar|Gradient Variance term (β=0)|More loss spikes during pruning|
|FANP - Recovery|Recovery tracker disabled|Slower convergence post-pruning|
|FANP - Adaptive|Fixed pruning rate|Catastrophic forgetting at high sparsity|
|**Full FANP**|All components enabled|Best accuracy-sparsity tradeoff|


# **10.  Advanced Extensions**

## **10.1  Meta-Learning α, β, γ**
Instead of hand-tuning the composite score weights, use a small meta-network trained on held-out tasks to predict optimal weights. This converts FANP into a learning-to-prune framework — a significant research contribution beyond the base project.

## **10.2  Continual Pruning**
Apply FANP to continual learning: as the model learns new tasks, the forgetting score naturally identifies which neurons are task-specific and which are shared. This prevents catastrophic forgetting in continual learning while also enabling compression — a dual contribution.

## **10.3  NAS Integration**
Use the forgetting score as a proxy metric for Neural Architecture Search: architectures where neurons show low forgetting under removal suggest naturally prunable designs. This connects FANP to zero-cost NAS proxies.

## **10.4  Hardware-Aware Forgetting**
Weight forgetting scores by hardware sensitivity: on edge TPUs, structured pruning is more valuable than unstructured. Extend the scheduler to maximize FLOPs reduction per unit forgetting, rather than absolute sparsity.


|<p>**★  Publication Target**</p><p>With strong CIFAR-10/ImageNet results, this project has publication potential at ICLR Workshop on Sparsity in Neural Networks or NeurIPS Workshop on Compression. The forgetting dynamics angle is genuinely underexplored in the pruning literature.</p>|
| :- |


***End of Project Document — FANP v1.0***
© 2025 Deep Learning Research Project	Page — end —
