# FANP Learning Guide: Phases 1–3
## From Basics to Advanced Understanding

*A comprehensive educational document for mastering neural network pruning with forgetting-aware dynamics.*

---

## Table of Contents

### Part 1: Foundations
1. [What Is a Neural Network](#part-1-foundations)
2. [What Is Neural Network Pruning](#what-is-neural-network-pruning)
3. [Why Do We Prune](#why-do-we-prune)
4. [Classic Pruning Methods](#classic-pruning-methods)

### Part 2: The Forgetting Problem
5. [What Is Forgetting in Neural Networks](#part-2-the-forgetting-problem)
6. [Why Existing Methods Fail](#why-existing-methods-fail)
7. [The Core Insight: Forgetting Score](#the-core-insight-forgetting-score)

### Part 3: Phase 1 — Baseline Training
8. [ResNet Architecture](#part-3-phase-1--baseline-training)
9. [Training Loop Mechanics](#training-loop-mechanics)
10. [Why Epoch 107 Matters](#why-epoch-107-matters)

### Part 4: Phase 2 — Importance Estimators
11. [Signal 1: Empirical Fisher Information](#part-4-phase-2--importance-estimators)
12. [Signal 2: Gradient Variance](#signal-2-gradient-variance)
13. [Signal 3: Loss Spike (Taylor Criterion)](#signal-3-loss-spike-taylor-criterion)
14. [Composing the Forgetting Score](#composing-the-forgetting-score)
15. [The Adaptive Pruning Scheduler](#the-adaptive-pruning-scheduler)

### Part 5: Phase 3 — Recovery and Structured Pruning
16. [Recovery Fine-Tuning](#part-5-phase-3--recovery-and-structured-pruning)
17. [Recovery Slope and OVERPRUNED Detection](#recovery-slope-and-overpruned-detection)
18. [Structured Pruning with Filter Removal](#structured-pruning-with-filter-removal)
19. [Putting It All Together](#putting-it-all-together)

### Appendix
20. [Mathematical Notation Reference](#appendix--mathematical-notation-reference)
21. [Code Walkthrough Examples](#code-walkthrough-examples)
22. [Debugging and Common Pitfalls](#debugging-and-common-pitfalls)

---

# PART 1: FOUNDATIONS

## What Is a Neural Network

A neural network is a mathematical function composed of many simple operations stacked together. Think of it like a pipeline:

```
Input Image          Hidden Layer 1         Hidden Layer 2        Output
(3 x 32 x 32)   ->  (64 neurons)       ->  (128 neurons)     ->  (10 classes)
  Raw pixels      Learn edges, colors    Learn shapes         Predict label
```

Each "neuron" (or more precisely, each weight and its associated computation) is responsible for a small part of the task. In a well-trained network:
- Early layers learn simple features (edges, colors)
- Middle layers learn complex features (wheels, faces, textures)
- Late layers make decisions based on those features

A trained ResNet-56 on CIFAR-10 has learned these features over 200 epochs of SGD. It achieves 92.90% accuracy on unseen test data.

### Key Intuition

Not all neurons are equally important. Some neurons are never used. Some neurons are used by many examples. Some neurons encode task-specific knowledge. The question "which neurons can we remove without hurting accuracy" is the core of pruning.

---

## What Is Neural Network Pruning

Pruning is the process of removing weights and neurons from a trained network to reduce its size and make it run faster.

### Types of Pruning

| Type | What Gets Removed | Speed | Accuracy Impact |
|---|---|---|---|
| **Weight pruning** | Individual weights → set to zero | Normal inference | Minimal, sparse tensor needed for speedup |
| **Channel/filter pruning** | Entire channels or filters removed | Actual speedup on regular hardware | Depends on importance estimation |
| **Layer pruning** | Entire layers removed | Huge speedup | High accuracy cost |

This project uses **structured filter pruning**: entire convolutional filters are physically removed, reducing both parameter count and actual computation.

### A Concrete Example

A conv layer has shape `Conv2d(64 input channels, 128 output channels, 3x3 kernel)`.

This creates: 64 * 128 * 3 * 3 = 73,728 parameters.

One filter is: 64 * 3 * 3 = 576 parameters (one output channel).

If we have 128 filters, we can rank them and remove the bottom 30% (38 filters). Remaining: 128 - 38 = 90 filters.

New layer size: 64 * 90 * 3 * 3 = 51,840 parameters.

Compression: 73,728 / 51,840 = 1.42x fewer weights in this layer.

---

## Why Do We Prune

### Problem 1: Model Size
A 92.90% accurate ResNet-56 takes ~3.5 MB of disk space. For mobile devices (smartphones, embedded systems), this is significant. Pruning reduces it to ~2.2 MB (40% reduction).

### Problem 2: Inference Speed
ResNet-56 on CIFAR-10 (32x32 images) runs in ~10 ms on edge devices. At scale (ImageNet, 224x224 images), it's much slower. Pruning reduces FLOPs: fewer filters = fewer multiplications = fewer clock cycles.

### Problem 3: Energy
On battery-powered devices, energy is proportional to the number of operations. Pruning reduces power consumption directly.

### The Tradeoff

We sacrifice accuracy slightly to gain speed and size reduction. Classical pruning loses:
- 30% sparsity (70% weights removed): -0.27% accuracy — acceptable
- 50% sparsity: -1.54% accuracy — still acceptable
- 70% sparsity: -11.12% accuracy — significant but maybe acceptable for speed
- 90% sparsity: -69.61% accuracy — collapse, unusable

FANP aims to recover accuracy at high sparsity by choosing which weights to remove more carefully.

---

## Classic Pruning Methods

### Method 1: Magnitude Pruning (Baseline)

**Idea:** If a weight's value is very small (close to zero), it probably doesn't matter much.

**Algorithm:**
1. Train model to convergence
2. For all weights w, compute |w|
3. Sort by magnitude
4. Remove the smallest X% by magnitude
5. Evaluate accuracy

**Intuition:** A weight of 0.001 or -0.003 is doing less work than a weight of 5.2, so remove it first.

**Problem:** Weight magnitude is static — it doesn't tell us about loss sensitivity. A small weight can still be critical if its gradient is large (sensitive).

**Code (simplified):**
```python
mask = (torch.abs(weights) > threshold)
weights_pruned = weights * mask
```

### Method 2: Hessian-Based Pruning (Optimal Brain Damage)

**Idea:** The Hessian matrix (second derivative of loss w.r.t. weights) tells us how removing a weight affects the final loss.

The diagonal approximation:
```
delta_loss ≈ (1/2) * H_ii * (delta_w_i)^2
```

where `H_ii` is the i-th diagonal element of the Hessian.

**Problem:** Computing the full Hessian is O(N^2) memory and O(N^2) time. For 855K parameters, that's ~730 billion values — impossible to store.

**Diagonal approximation:** Assume off-diagonal elements are zero. This is fast but inaccurate.

### Method 3: Gradient-Based Estimators

**Idea:** Instead of weight magnitude, use the gradient. A weight with large gradient is sensitive to change.

**Formula:**
```
Importance_i ∝ |grad_i|
```

**Problem:** Gradients are noisy and change every step. A single backward pass gives an unreliable snapshot.

**Fisher Information (improvement):** Average the gradient signal:
```
F_hat_i = (1/N) * sum over batches ( grad_i^2 )
```

This is stable and interpretable as the expected squared gradient.

---

# PART 2: THE FORGETTING PROBLEM

## What Is Forgetting in Neural Networks

"Forgetting" in this context means: when we remove a neuron, by how much does the model's performance drop?

### Example: What Happens When We Remove a Neuron

Suppose we have a trained ResNet-56 with validation accuracy 93.24%.

We remove (prune) some filters from layer 10. Now:
- With finetuning: accuracy might drop to 92.5% for a moment, then recover to 93.0% after 500 training steps
- Without finetuning: accuracy might recover to 92.8% after 500 steps
- In a bad case: accuracy does NOT recover — it stays at 80% even after 500 steps → we removed something critical

**Forgetting rate** = how bad the initial drop is = how much the model "forgets" what that neuron was doing.

**Recovery slope** = how fast accuracy comes back = how quickly the model re-learns to work without that neuron.

---

## Why Existing Methods Fail

### Failure Mode 1: Silent Forgetting

Magnitude pruning ranks weights by size. But small weights can be critical:

```
Weight w_A = 0.0002 (very small)  BUT  grad_A = 100.0 (very large)
Effect: w_A * grad_A = 0.02  ← this weight contributes significantly to gradients

Weight w_B = 2.5 (large)  BUT  grad_B = 0.0001 (tiny)
Effect: w_B * grad_B = 0.00025  ← this weight is not doing much

Magnitude ranking: Remove w_A first (it's smaller)
Reality: w_A is more important; we should remove w_B
```

Result: Silent forgetting — early on, the model seems fine, but downstream, losses accumulate.

### Failure Mode 2: Catastrophic Interference

If we remove multiple co-adapted neurons at once:

```
Neuron A detects "wheels" with output = 0.7
Neuron B detects "round shapes" with output = 0.5
They feed into a layer that says: if (A=0.7 AND B=0.5) → car=True

If we remove BOTH A and B, the downstream logic collapses suddenly.
Their individual importance might be low, but combined removal is catastrophic.
```

Magnitude ranking handles each weight independently. It doesn't see co-adaptation.

### Failure Mode 3: Static vs. Dynamic Importance

A weight's importance can change based on what the model is learning in this training batch.

```
Batch 1: mostly dogs → neuron X (fur detector) is critical
Batch 2: mostly cars → neuron Y (wheel detector) is critical

Single backward pass captures only the current snapshot.
Averaging over K steps (Fisher) helps, but we still don't see pruning consequences.
```

---

## The Core Insight: Forgetting Score

Instead of estimating "will this weight be important," we measure "what happens when we remove it."

### Three Information Sources

#### Source 1: Historical Sensitivity (Fisher)

"How much has this weight been adjusted by past updates?"

```
F_hat_i = (1/K) * sum over last K batches ( grad_i^2 )
```

**Intuition:** If a weight's gradient is consistently large (over K batches), it's being actively used.

**Example:**
- Weight in layer that stabilizes training: grad_i = [2.1, 1.8, 2.3, ...] → high F_hat
- Weight in a dead neuron: grad_i = [0.0001, 0.0001, 0.0001, ...] → low F_hat

#### Source 2: Recent Volatility (Gradient Variance)

"How noisy is this weight's gradient right now?"

```
sigma^2(g_i) = Var( {grad_i over last K steps} )
```

**Intuition:** High variance means the weight is contested — different training examples want it to do different things. This suggests it encodes task-specific knowledge.

**Example:**
- Neuron encoding "dog ears": gets strong positive gradients from dog examples, strong negative from cat examples → high variance
- Neuron encoding "generic low-level edge": consistent small positive gradient → low variance

A high-variance weight = important for the diversity of the task.

#### Source 3: Direct Loss Impact (Taylor Criterion)

"If we remove this weight right now, how much would loss increase?"

Using first-order Taylor expansion:
```
delta_L_approx ≈ |grad_i * weight_i|  (actually (grad_i * weight_i)^2 for direction-invariance)
```

**Why this formula?**

Starting from L(w), if we move w → w + delta_w, the change in loss is approximately:

```
L(w + delta_w) ≈ L(w) + grad * delta_w  + (1/2) * H * delta_w^2 + ...
```

Setting w_i → 0 means delta_w_i = -w_i:

```
delta_L ≈ grad_i * (-w_i) = -grad_i * w_i
```

Taking the square (because we care about magnitude, not direction):

```
delta_L_approx = (grad_i * w_i)^2
```

**Intuition:** This is measured, not estimated. We actually know how much removing this weight hurts.

**Problem:** Computing this for every weight is expensive (requires multiple forward passes). The Taylor approximation lets us compute it in one backward pass.

### Composing the Signals

The Forgetting Score combines all three:

```
S_i = alpha * F_hat_i  +  beta * sigma^2(g_i)  +  gamma * delta_L_i
     = 0.5  * F_hat_i  +  0.3  * sigma^2(g_i)  +  0.2  * delta_L_i
```

**Interpretation:**
- High S_i: weight is historically important (high F), contested (high sigma^2), and removing it would hurt loss (high delta_L) → DO NOT PRUNE
- Low S_i: weight is unimportant (low F), stable (low sigma^2), and removing it barely affects loss (low delta_L) → PRUNE FIRST

**Why these weights?** In practice, discovery through many ablations. Typical pattern: Fisher is most important, gradient variance second, Taylor third.

---

# PART 3: PHASE 1 — BASELINE TRAINING

## ResNet Architecture

ResNet-56 is a residual network designed for CIFAR-10. The "56" refers to the number of layers (including the initial conv1 and final FC).

### High-Level Structure

```
Input: 32x32 RGB image (CIFAR-10 standard)
      |
      v
conv1: 3 → 64 channels, 3x3 kernel, stride=1
      |
      v
block1 (18 residual blocks, 64 channels each)
      |
      v
block2 (18 residual blocks, 128 channels each, stride=2 downsampling)
      |
      v
block3 (18 residual blocks, 256 channels each, stride=2 downsampling)
      |
      v
avgpool: reduce spatial dims to 1x1
      |
      v
fc: 256 → 10 classes
      |
      v
Output: logits for [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
```

Total: 18*3 = 54 residual blocks + (conv1 + fc) = 56 layers.
Parameter count: ~855K.

### Residual Block Details

A residual block (bottleneck):

```
Input x (e.g., shape [batch=128, channels=64, H=32, W=32])
  |
  v
conv1: 3x3, same channels (e.g., 64 → 64)
  |
  v
ReLU
  |
  v
conv2: 3x3, same channels (e.g., 64 → 64)
  |
  v
+ (add) shortcut connection (the input x)
  |
  v
ReLU
  |
  v
Output: same shape as input
```

**Why residuals?**

Without the shortcut, gradients propagate through many layers and get very small (vanishing gradient). With the shortcut, gradients can "skip" over blocks, stabilizing training.

**In code:**
```python
def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.relu(out)
    out = self.conv2(out)
    out = out + identity  # <-- the skip connection
    out = self.relu(out)
    return out
```

---

## Training Loop Mechanics

Phase 1 training uses standard supervised learning:

```python
for epoch in range(200):
    for batch in train_loader:
        images, labels = batch
        
        # Forward pass
        logits = model(images)
        loss = cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val_loss:
        save(model, 'resnet56_best.pth')
        best_val_loss = val_loss
```

### Key Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Learning rate schedule | CosineAnnealingLR | Smooth decay, better convergence than step-wise |
| Initial LR | 0.1 | Standard for SGD + momentum |
| Momentum | 0.9 | Helps with oscillation, standard for CNNs |
| Weight decay | 5e-4 | L2 regularization, prevents overfitting |
| Batch size | 128 | tradeoff between gradient noise and memory |
| Epochs | 200 | Standard for CIFAR-10; longer = better generalization |
| Optimizer | SGD | Simple, stable, effective |

### Why CosineAnnealingLR?

```
Learning rate schedule over 200 epochs:

LR(epoch) = 0.1 * (1 + cos(pi * epoch / 200)) / 2

Visualized:
0.10  |     *
0.08  |    * *
0.06  |   *   *
0.04  |  *     *
0.02  | *       *
0.00  |*---------* 
      0        200

Key property: smooth, continuous decay without sharp drops.
Effect: In final epochs, LR → 0, so weights stabilize and generalize well.
```

---

## Why Epoch 107 Matters

Training results:

```
Epoch   Train Loss  Val Loss  Val Acc  Checkpoint
-----   ----------  --------  -------  ----------
100     0.112       0.245     93.05%
105     0.108       0.240     93.18%
106     0.107       0.238     93.21%
107     0.106       0.236     93.24%   <-- BEST (saved as resnet56_best.pth)
108     0.105       0.239     93.22%   (started overfitting)
...
150     0.045       0.310     92.95%   (test acc dropping)
...
200     0.012       0.420     92.90%   <-- LAST (saved as resnet56_last.pth)
```

### The Key Insight

After epoch 107, the model still learns on the training set (train loss keeps decreasing from 0.106 → 0.012), BUT validation accuracy starts dropping (93.24% → 92.90%).

This is **overfitting:** the model memorizes training data details that don't generalize.

### Why Not Use the Last Checkpoint?

If we pruned from epoch 200 (92.90% accuracy), the pruning baseline is weaker. Magnitude pruning would then appear less of a problem because we're starting from an already-weakened model. That's unfair comparison.

**Solution:** Always prune from the best validation checkpoint (epoch 107, 93.24% accuracy). This gives classical methods their best chance and makes our improvements more meaningful.

---

# PART 4: PHASE 2 — IMPORTANCE ESTIMATORS

## Signal 1: Empirical Fisher Information

Fisher Information Matrix (FIM) measures the curvature of the loss landscape.

### Intuition via a 1D Example

Imagine loss as a 1D curve:

```
Loss
  |
  |    *
  |   * *
  |  *   *
  | *     *
  |*       *
  +----------- Weight
 -5  0  5

A steep region: small change in weight → big change in loss
A flat region: weight can shift a lot without affecting loss much
```

The slope at any point is the gradient. The curvature (how fast slope changes) is related to the Hessian.

Diagonal of the Hessian ≈ how "important" a weight is for loss.

### Definition

For a batch of N samples:

```
F_hat_i = (1/N) * sum_{k=1}^{N} (∂L_k / ∂w_i)^2
```

where L_k is the loss for the k-th sample.

### Intuition: Why Square the Gradient?

```
Gradient measures slope.
Squared gradient (Fisher) measures "how much does loss change if we perturb w_i"

High square gradient = small change in weight → big change in loss → important weight
Low square gradient = weight can change freely → unimportant
```

### Computing Fisher

**Naive approach:** For each weight, compute its gradient over N batches, square, average.

**Smart approach:** Hook into the backward pass to capture all gradients in one pass.

```python
def compute_fisher(model, loader, num_batches=50):
    fisher = {}
    
    # Register backward hooks to capture gradients
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    def make_hook(name):
        def hook(grad):
            fisher[name] += grad ** 2
        return hook
    
    # Attach hooks
    for name, param in model.named_parameters():
        param.register_hook(make_hook(name))
    
    # Run forward/backward on batches
    model.eval()
    with torch.no_grad():  # don't need autograd for second derivatives
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            
            logits = model(images)
            loss = cross_entropy(logits, labels)
            loss.backward()
    
    # Average over batches
    for name in fisher:
        fisher[name] /= num_batches
    
    return fisher
```

### Fisher Interpretation

```
Large Fisher (F_hat_i = 0.5+):
- This weight's gradient is consistently large
- Loss is very sensitive to changes
- Example: a core classification feature

Medium Fisher (F_hat_i = 0.01 - 0.1):
- This weight contributes but not critically
- Loss sensitivity is moderate

Small Fisher (F_hat_i < 0.0001):
- This weight barely affects loss
- Likely candidates for pruning
```

---

## Signal 2: Gradient Variance

While Fisher captures the magnitude of gradients, it doesn't capture their consistency.

### Definition

Over K batches, compute the variance of gradients:

```
sigma^2(g_i) = Var( {grad_i^(1), grad_i^(2), ..., grad_i^(K)} )
              = (1/K) * sum_{k=1}^{K} (grad_i^(k) - mean(grad_i))^2
```

### Intuition: What Does High Variance Mean?

**Example 1: Dedicated Dog Detector Neuron**

```
Batch 1: dog images  → large positive gradient (grad = +5.0)
Batch 2: cat images  → large negative gradient (grad = -4.5)
Batch 3: dog images  → large positive gradient (grad = +4.8)
Batch 4: cat images  → large negative gradient (grad = -5.1)

Mean gradient ≈ 0 (positive and negative cancel out)
Variance = high (gradients swing from -5 to +5)

Interpretation: This neuron is critical — it encodes something the model actively disagrees about for different inputs.
```

**Example 2: Generic Background Blur Detector**

```
Batch 1: any images  → tiny positive gradient (grad = +0.01)
Batch 2: any images  → tiny positive gradient (grad = +0.01)
Batch 3: any images  → tiny positive gradient (grad = +0.01)
Batch 4: any images  → tiny positive gradient (grad = +0.01)

Mean gradient ≈ 0.01
Variance = very low (all gradients close to +0.01)

Interpretation: This neuron is consistent but marginal — not task-specific.
```

### Computing Gradient Variance

```python
def compute_gradient_variance(model, loader, window_size=50):
    grad_history = {}
    variance = {}
    
    for name, param in model.named_parameters():
        grad_history[name] = []
        
    def make_hook(name):
        def hook(grad):
            grad_history[name].append(grad.detach().clone())
            if len(grad_history[name]) > window_size:
                grad_history[name].pop(0)  # Keep only last K
        return hook
    
    # Attach hooks
    for name, param in model.named_parameters():
        param.register_hook(make_hook(name))
    
    # Collect gradients
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        logits = model(images)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute variance
    for name in grad_history:
        grads = torch.stack(grad_history[name])  # [K, *shape]
        variance[name] = torch.var(grads, dim=0)  # variance per element
    
    return variance
```

---

## Signal 3: Loss Spike (Taylor Criterion)

If we remove a weight (set it to zero), by how much does loss increase?

### Exact Formula

Starting from loss L(w), if we set weight w_i → 0, the change is:

```
delta_L = L(..., w_i=0, ...) - L(..., w_i=orig, ...)
```

Computing this exactly requires a forward pass for every weight — 855K forward passes on a ResNet-56. Impossible in practice.

### Taylor Approximation

Use the first-order Taylor expansion:

```
L(..., w_i + delta_w_i, ...) ≈ L(..., w_i, ...) + (∂L/∂w_i) * delta_w_i

Setting w_i → 0 means delta_w_i = -w_i:

delta_L ≈ (∂L/∂w_i) * (-w_i) = -(∂L/∂w_i) * w_i

Taking magnitude (because we care about impact, not direction):

|delta_L| ≈ |(∂L/∂w_i) * w_i|

Or squared (more stable numerically):

delta_L_approx = ((∂L/∂w_i) * w_i)^2
```

### Code Implementation

```python
def compute_loss_spike(model, loader, num_batches=50):
    loss_spike = {}
    
    for name, param in model.named_parameters():
        loss_spike[name] = torch.zeros_like(param)
    
    model.eval()
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        images.requires_grad = False
        logits = model(images)
        loss = cross_entropy(logits, labels)
        
        # Backward to get gradients
        loss.backward()
        
        # Compute Taylor criterion per weight
        for name, param in model.named_parameters():
            if param.grad is not None:
                loss_spike[name] += (param.grad * param) ** 2
    
    # Average
    for name in loss_spike:
        loss_spike[name] /= num_batches
    
    return loss_spike
```

### Why Is This Better Than Exact?

```
Exact loss spike (intractable):
  for w_i in all_weights:
      delta_L[i] = forward_pass(remove w_i) - baseline_loss  <-- 855K forward passes

Taylor approximation (one backward pass):
  delta_L[i] ≈ (grad[i] * w[i])^2  <-- one backward pass
```

Cost: 855K forward passes vs. 1 backward pass = **>1000x speedup** with good accuracy.

Error: The approximation assumes the loss surface is roughly quadratic locally, which is often true near a minimum.

---

## Composing the Forgetting Score

Now we have three signals:

1. **Fisher F_hat_i:** historical sensitivity
2. **Gradient Variance sigma^2(g_i):** current volatility
3. **Loss Spike delta_L_i:** direct impact

### The Composite Formula

```
S_i = alpha * F_hat_i  +  beta * sigma^2(g_i)  +  gamma * delta_L_i
```

With default weights:
```
alpha = 0.5    (Fisher is most predictive)
beta = 0.3     (Gradient variance is secondary)
gamma = 0.2    (Loss spike is supplementary)
```

### Normalization

Before combining, each signal should be normalized to the same scale:

```python
def compute_forgetting_score(fisher, grad_var, loss_spike, alpha=0.5, beta=0.3, gamma=0.2):
    # Normalize each signal to [0, 1] per-layer
    fisher_norm = {}
    grad_var_norm = {}
    loss_spike_norm = {}
    
    for name in fisher:
        f_min, f_max = fisher[name].min(), fisher[name].max()
        fisher_norm[name] = (fisher[name] - f_min) / (f_max - f_min + 1e-8)
        
        g_min, g_max = grad_var[name].min(), grad_var[name].max()
        grad_var_norm[name] = (grad_var[name] - g_min) / (g_max - g_min + 1e-8)
        
        l_min, l_max = loss_spike[name].min(), loss_spike[name].max()
        loss_spike_norm[name] = (loss_spike[name] - l_min) / (l_max - l_min + 1e-8)
    
    # Composite score
    forgetting_score = {}
    for name in fisher:
        forgetting_score[name] = (
            alpha * fisher_norm[name] +
            beta * grad_var_norm[name] +
            gamma * loss_spike_norm[name]
        )
    
    return forgetting_score
```

### Interpretation: Ranking Weights for Pruning

```
bucket = []
for layer_name in all_layers:
    for weight_idx in all_weights_in_layer[layer_name]:
        score = forgetting_score[layer_name][weight_idx]
        bucket.append((layer_name, weight_idx, score))

# Sort ascending (low score first)
bucket.sort(key=lambda x: x[2])

# Weights with lowest scores are candidates for pruning
candidates_to_prune = bucket[:len(bucket) // 3]  # Bottom 33%
```

Low score = not historically sensitive (low Fisher) + not volatile (low grad_var) + not directly impactful (low loss_spike).

These weights are safely removable.

---

## The Adaptive Pruning Scheduler

The core idea: pruning rate should adapt to how "healthy" the model is.

If the model is still forgetting a lot (average forgetting score is high), we should slow down pruning to let it stabilize.

### Algorithm

```python
class AdaptivePruningScheduler:
    def __init__(self, base_rate=0.10, tau=0.05):
        self.base_rate = base_rate      # baseline pruning rate
        self.current_rate = base_rate   # updated per round
        self.tau = tau                  # threshold for mean forgetting score
    
    def get_next_rate(self, forgetting_scores):
        """
        Input: forgetting_scores (dict of tensors per layer)
        Output: pruning_rate for this round
        """
        # Compute global mean forgetting score
        all_scores = torch.cat([s.flatten() for s in forgetting_scores.values()])
        mean_fs = all_scores.mean().item()
        
        # Adaptive logic
        if mean_fs > self.tau:
            # Model is forgetting too much, slow down
            self.current_rate = self.current_rate / 2.0
        else:
            # Model is stable, restore rate
            self.current_rate = min(self.current_rate * 1.2, self.base_rate)
        
        return self.current_rate, mean_fs
```

### Intuition

```
Round 1: prune 10% (rate=0.10) of filters
  -> mean_fs = 0.06 (slightly high)

Round 2: mean_fs > tau, so halve rate
  -> rate = 0.05 (prune 5% instead)
  -> after fine-tuning, model recovers well

Round 3: mean_fs = 0.03 (lower), so increase
  -> rate = 0.06

Round 4: mean_fs = 0.02 (low), so increase
  -> rate = 0.072 (approaching base_rate)
```

This prevents "catastrophic interference" by slowing down when the model is unstable.

---

# PART 5: PHASE 3 — RECOVERY AND STRUCTURED PRUNING

## Recovery Fine-Tuning

After removing filters, the model's accuracy drops. Recovery fine-tuning is a short training phase to let the remaining weights adapt and regain accuracy.

### Setup

```python
class FineTuner:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def fine_tune(self, train_loader, val_loader, n_steps=500, lr=0.005, eval_interval=50):
        """
        Args:
            train_loader: training data
            val_loader: validation data
            n_steps: how many optimization steps to run
            lr: learning rate (lower than initial training to not disturb surviving weights)
            eval_interval: evaluate every N steps
        
        Returns:
            RecoveryTrace with accuracy history
        """
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        
        initial_val_acc = self.evaluate(val_loader)
        step_accuracies = []
        
        for step in range(n_steps):
            # Train one step
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # Re-apply pruning masks
                # (This line is critical: after sgd updates, some zeroed weights might get non-zero values again)
                self.apply_masks()
                
                break  # Only one batch per step (for speed)
            
            # Evaluate periodically
            if step % eval_interval == 0:
                val_acc = self.evaluate(val_loader)
                step_accuracies.append((step, val_acc))
        
        final_val_acc = self.evaluate(val_loader)
        
        return RecoveryTrace(
            step_acc=step_accuracies,
            initial_acc=initial_val_acc,
            final_acc=final_val_acc
        )
    
    def apply_masks(self):
        """Re-apply pruning masks so zeroed weights stay zero."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight.data *= module.weight_mask
    
    def evaluate(self, val_loader):
        """Compute validation accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                predicted = logits.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        self.model.train()
        return 100.0 * correct / total
```

### Key Details

#### 1. Lower Learning Rate

Initial training uses LR=0.1. Fine-tuning uses LR=0.005 (20x lower).

Why? The network is already mostly-trained. A high learning rate would overwrite the knowledge of surviving weights. A low learning rate lets surviving weights adjust minimally while the pruned-away computation is redistributed.

#### 2. Mask Re-application

```python
# After optimizer.step(), weights have changed
# Some originally-zero weights (in pruned filters) might be non-zero now

module.weight.data *= module.weight_mask  # Zero them out again

# Now pruned = 0, others can vary
```

This ensures pruned weights never contribute to the forward pass.

#### 3. RecoveryTrace Dataclass

```python
@dataclass
class RecoveryTrace:
    step_acc: List[Tuple[int, float]]     # [(step, accuracy), ...]
    initial_acc: float                     # Before fine-tuning
    final_acc: float                       # After fine-tuning
    
    @property
    def recovery_slope(self):
        """Accuracy gain per step."""
        if len(self.step_acc) < 2:
            return 0.0
        steps = [x[0] for x in self.step_acc]
        accs = [x[1] for x in self.step_acc]
        delta_acc = accs[-1] - accs[0]
        delta_steps = steps[-1] - steps[0]
        return delta_acc / (delta_steps + 1e-8)
```

---

## Recovery Slope and OVERPRUNED Detection

### Recovery Slope

After fine-tuning, we compute:

```
recovery_slope = (accuracy_after - accuracy_before) / n_steps
```

Units: percentage points per step.

### Interpretation

| Slope | Interpretation |
|---|---|
| > 0.2 %/step | Excellent recovery, pruned weights were less important |
| 0.1 – 0.2 %/step | Good recovery, stable |
| 0.0 – 0.1 %/step | Slow recovery, we removed some important things |
| < 0 %/step | **OVERPRUNED**: accuracy is still dropping, critical info was removed |

### OVERPRUNED Flag

```python
class RecoveryMetrics:
    round_idx: int
    sparsity: float
    acc_before: float
    acc_after: float
    recovery_slope: float
    overpruned: bool  # True if recovery_slope < slope_threshold
```

In the code:

```python
tracker = RecoveryTracker(slope_threshold=0.005)  # threshold: 0.5 %/step

metrics = tracker.measure(train_loader, val_loader, round_idx=3, sparsity=0.27)
# metrics = RecoveryMetrics(..., recovery_slope=0.0023, overpruned=True)

# Meaning: we're recovering at 0.23%/step, which is below the 0.5%/step threshold
# Signal: next round should prune more cautiously
```

### Using the Signal

The adaptive scheduler uses recovery_slope indirectly:

```python
# After round N:
metrics = tracker.measure(...)
if metrics.overpruned:
    print("WARNING: OVERPRUNED at round", metrics.round_idx)
    # Could halt pruning or reduce rate further

# The scheduler also checks mean_fs to adapt rate:
mean_fs = compute_mean_forgetting_score(model, loader)
if mean_fs > tau:
    scheduler.rate = scheduler.rate / 2  # Reduce next round's pruning
```

---

## Structured Pruning with Filter Removal

### Why Structured Pruning?

**Unstructured:** Set individual weights to zero.
- Pro: Fine-grained control, highest potential accuracy
- Con: Requires sparse tensor support to actually run faster; most hardware doesn't have this

**Structured:** Remove entire filters (output channels of a conv layer).
- Pro: Actual speedup on standard hardware, easier to deploy
- Con: Coarser granularity, might remove slightly-important filters alongside useless ones

This project uses structured pruning for real-world applicability.

### How to Score Filters?

We have per-weight forgetting scores `S_i` for all 855K weights. We need to aggregate them to per-filter scores.

A filter in a conv layer has shape: `[input_channels, kernel_height, kernel_width]`

For example, filter #42 in a `Conv2d(64, 128, 3x3)` layer has 64*3*3 = 576 weights.

**Aggregation:** Average the scores of all weights in the filter.

```
I_filter_j = mean(S_i for all i in filter j)
```

Then rank filters by importance and remove the bottom X% by importance.

### Dependency Graph (torch-pruning)

Simply removing a filter breaks the dependency structure:

```
Layer 1 output: Shape [batch, 128, H, W]   <-- 128 filters
Layer 2 input:  Shape [batch, 128, H, W]   <-- expects 128 filters

If we remove filter #42 from layer 1, we must also remove the corresponding input channel from layer 2
```

The torch-pruning library builds a **dependency graph** to track this:

```
Layer1.conv.weight    -> Layer1 is a producer
                      |
                      v
Layer2.conv.weight    -> Layer2 is a consumer (depends on Layer1 output)
        |
        v
Layer2.bias.weight    -> Layer2.bias depends on Layer2.conv
```

When we remove a filter, torch-pruning automatically removes dependent structures.

### Code Walkthrough

```python
class StructuredFANPPruner:
    def __init__(self, model, importance_estimator):
        self.model = model
        self.importance_estimator = importance_estimator
        
        # Build dependency graph
        example_input = torch.randn(1, 3, 32, 32).to(model.device)
        self.dep_graph = tp.DependencyGraph()
        self.dep_graph.build_dependency(self.model, example_input)
    
    def prune(self, loader, target_sparsity=0.3, example_inputs=None):
        """
        Remove bottom target_sparsity fraction of filters.
        """
        # Step 1: Compute forgetting scores
        scores = self.importance_estimator.score_layers()
        
        # Step 2: Aggregate to filter level
        filter_importance = {}
        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                filter_importance[module_name] = self._aggregate_to_filters(
                    scores, module_name, module
                )
        
        # Step 3: Rank and select for removal
        pruning_plan = {}
        for module_name, importance_values in filter_importance.items():
            n_filters = importance_values.shape[0]
            n_to_prune = int(n_filters * target_sparsity)
            
            # Sort index: most important first
            _, indices = torch.topk(importance_values, n_filters - n_to_prune, largest=True)
            
            pruning_plan[module_name] = indices.cpu().numpy()
        
        # Step 4: Execute pruning via dependency graph
        # (torch-pruning handles the complex logic)
        pruner = tp.MetaPruner(
            self.model,
            importance=filter_importance,  # our scores
            iterative_steps=1,
            ch_sparsity=target_sparsity
        )
        
        # Apply pruning
        pruner.step(interactive=False)
        
        # Step 5: Measure and return
        pruned_params = count_parameters(self.model)
        return {
            'pruned_params': pruned_params,
            'sparsity': current_sparsity(self.model),
            'pruning_plan': pruning_plan
        }
    
    def _aggregate_to_filters(self, scores, module_name, module):
        """
        Given per-weight scores, compute per-filter scores.
        Filter i = average of all weights in ouput channel i
        """
        weight_scores = scores[module_name + '.weight']  # [out_ch, in_ch, kH, kW]
        
        # Average across input channels, kernel size
        filter_scores = weight_scores.mean(dim=(1, 2, 3))  # [out_ch]
        
        return filter_scores
```

### Example: Before and After

**Before pruning (30% target sparsity):**
- Layer "layer1.1.conv1": 64 input channels, 64 output channels
  - Weights shape: [64, 64, 3, 3] = 36,864 parameters
- All filters have scores, ranked by importance

**Pruning action:**
- Compute filter importance for all 64 filters
- Remove bottom 30% = bottom 19 filters
- Remaining: 64 - 19 = 45 filters

**After pruning:**
- Layer "layer1.1.conv1": 64 input channels, 45 output channels
  - Weights shape: [45, 64, 3, 3] = 27,648 parameters
  - Reduction: 36,864 - 27,648 = 9,216 fewer parameters (25% reduction in this layer)

**Cascading effect via dependency graph:**
- The next layer (layer1.2.conv1) now receives only 45 input channels (not 64)
- Its input is automatically adjusted
- Its computation reduces from [128, 64, 3, 3] to [128, 45, 3, 3]

Torch-pruning handles all of this automatically through the dependency graph.

---

## Putting It All Together

### The Full FANP Pruning Pipeline

```
1. START with trained ResNet-56 (epoch 107 checkpoint)
   accuracy: 93.24%

2. SET target sparsity to 30% (will recursively prune until reaching this)

3. INITIALIZE:
   - ForgettingScore estimator
   - AdaptivePruningScheduler
   - StructuredFANPPruner
   - RecoveryTracker

4. FOR each pruning round (1, 2, 3, ...):
   
   a. Compute forgetting score for all weights
      - EmpiricalFisher over last 50 train batches
      - GradientVariance over sliding window K
      - LossSpike (Taylor criterion) per weight
      - Composite: S = 0.5*F + 0.3*V + 0.2*L
   
   b. Obtain adaptive pruning rate
      - Get mean forgetting score
      - If mean_fs > tau: rate = rate / 2
      - Otherwise: rate = min(rate * 1.2, base_rate)
   
   c. Run StructuredFANPPruner
      - Aggregate per-weight scores to per-filter scores
      - Build dependency graph
      - Remove bottom `rate` fraction of filters
      - Current sparsity increases (e.g., 10.0% -> 19.0%)
   
   d. Run FineTuner for recovery
      - Fine-tune for n_steps = 500 (or n_steps = 20 in quick mode)
      - Use learning rate = 0.005 (low)
      - Eval every 100 steps
      - Re-apply masks after each step
   
   e. Measure recovery via RecoveryTracker
      - Compute recovery_slope = delta_acc / delta_step
      - Flag OVERPRUNED if slope < 0.005
      - Record metrics: round_idx, sparsity, acc_before, acc_after, slope
   
   f. Check termination
      - If current_sparsity >= target_sparsity: STOP
      - Else: continue to next round

5. FINAL EVALUATION
   - Evaluate on test set
   - Record final accuracy
   - Compare against Magnitude and Magnitude+FT baselines
```

### Execution Example (30% target sparsity)

```
TARGET SPARSITY: 30%

Round 1:
  mean_FS = 0.0467, rate=0.100 (unchanged)
  Remove 10.0% filters -> sparsity 9.962%
  FineTuner: 90.60% -> 92.74% (+2.14%) slope=0.107%/step OK
  
Round 2:
  mean_FS = 0.0362, rate=0.100 (unchanged)
  Remove 9.1% filters -> sparsity 18.929%
  FineTuner: 87.74% -> 91.84% (+4.10%) slope=0.205%/step OK
  
Round 3:
  mean_FS = 0.0246, rate=0.100 (unchanged)
  Remove 8.5% filters -> sparsity 26.998%
  FineTuner: 70.86% -> 91.20% (+20.34%) slope=1.017%/step OK
  
Check: 26.998% >= 30%? No.

Round 4:
  mean_FS = 0.0189, rate=0.100 (unchanged)
  Remove 3.2% more filters -> sparsity 29.8%
  FineTuner: 72.40% -> 91.80% (+19.40%) slope=1.94%/step OK

Check: 29.8% >= 30%? YES -> TERMINATE

Final Test Accuracy: 90.45%
Comparison:
  - Magnitude (same sparsity): 92.61%
  - FANP: 90.45%
  - Difference: -2.16% (shows FANP uses different criteria, not always better at 30% —
    might be better at higher sparsity)
```

---

# APPENDIX — MATHEMATICAL NOTATION REFERENCE

## Notation

| Symbol | Meaning | Example |
|---|---|---|
| w_i | The i-th weight in the network | w_523 = 0.152 |
| ∂L/∂w_i | Gradient of loss with respect to w_i | 2.3 (loss increases if w_i increases) |
| F_hat_i | Empirical Fisher value for weight i | 0.45 (average squared gradient) |
| σ^2(g_i) | Variance of gradients for weight i | 0.012 (gradients swing from -0.1 to +0.1) |
| δL_i | Loss spike (Taylor approx) for weight i | 0.0089 (loss increase if w_i → 0) |
| S_i | Forgetting Score (composite) for weight i | 0.38 (normalized 0-1 scale) |
| τ | Threshold for adaptive scheduler | 0.05 (if mean_fs > 0.05, slow down) |
| α, β, γ | Weights for Forgetting Score components | α=0.5, β=0.3, γ=0.2 |
| N | Batch size or number of samples | 128 |
| K | Window size for averaging | 50 batches |
| L | Cross-entropy loss (classification) | 0.23 (lower is better) |
| ŷ | Model prediction logits | [0.1, 2.3, -0.5, ...] (10-dim for CIFAR-10) |
| y | Ground truth label | 2 (cat) |

## Key Formulas

### Empirical Fisher
```
F_hat_i = (1/N) * sum_{k=1}^{N} (∂L_k/∂w_i)^2
```

Interpretation: Average squared gradient = parameter sensitivity.

### Gradient Variance
```
σ^2(g_i) = Var({grad_i^(1), ..., grad_i^(K)})
         = (1/K) * sum_{k=1}^{K} (grad_i^(k) - mean(grad_i))^2
```

Interpretation: Variance of gradient history = task-specificity.

### Loss Spike (Taylor)
```
δL_i ≈ (∂L/∂w_i * w_i)^2
```

Interpretation: Squared gradient-weight product = first-order loss impact of removal.

### Forgetting Score
```
S_i = α * F_hat_i + β * σ^2(g_i) + γ * δL_i
```

where α + β + γ = 1. Default: α=0.5, β=0.3, γ=0.2.

### Recovery Slope
```
recovery_slope = (accuracy_final - accuracy_initial) / n_steps
```

Units: percentage points per optimization step.

### Sparsity Ratio
```
sparsity = 1 - (pruned_params / original_params)
```

Example: 855,770 params → 536,844 params → sparsity = 1 - 536,844/855,770 ≈ 0.37 (37% sparse = 63% remaining).

---

# CODE WALKTHROUGH EXAMPLES

## Example 1: Computing Fisher Information

```python
# Step-by-step Fisher computation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_fisher(model, train_loader, device='cuda', num_batches=50):
    """
    Compute empirical Fisher matrix (diagonal approximation).
    
    Fisher_ii = E[(dL/dw_i)^2]
    
    Args:
        model: trained neural network
        train_loader: training data
        device: 'cuda' or 'cpu'
        num_batches: how many batches to average over
    
    Returns:
        fisher: dict {param_name: tensor of Fisher values}
    """
    
    model.eval()  # Set to evaluation mode (no dropout, batchnorm frozen)
    
    # Initialize accumulator for sum of squared gradients
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    batch_count = 0
    
    for images, labels in train_loader:
        if batch_count >= num_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad ** 2
        
        # Zero gradients for next iteration
        model.zero_grad()
        
        batch_count += 1
    
    # Average over batches
    for name in fisher:
        fisher[name] /= batch_count
    
    return fisher

# Usage:
# model = ResNet56(num_classes=10).to('cuda')
# train_loader = DataLoader(train_dataset, batch_size=128)
# fisher = compute_fisher(model, train_loader)
# print(fisher['layer1.0.conv1.weight'].shape)  # [64, 64, 3, 3]
# print(fisher['layer1.0.conv1.weight'].mean())  # ~0.23
```

## Example 2: Computing Gradient Variance

```python
import torch
import torch.nn as nn
from collections import defaultdict

class GradientVarianceTracker:
    """Track gradient variance over a sliding window of K steps."""
    
    def __init__(self, model, window_size=50):
        self.model = model
        self.window_size = window_size
        
        # Initialize history
        self.grad_history = {}
        for name, param in model.named_parameters():
            self.grad_history[name] = []
    
    def step(self):
        """Call after loss.backward() and before optimizer.step()."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Store gradient
                grad_copy = param.grad.detach().clone()
                self.grad_history[name].append(grad_copy)
                
                # Keep only last K
                if len(self.grad_history[name]) > self.window_size:
                    self.grad_history[name].pop(0)
    
    def get_variance(self):
        """
        Compute current variance per parameter.
        
        Returns:
            variance: dict {param_name: tensor}
        """
        variance = {}
        
        for name in self.grad_history:
            grads = self.grad_history[name]
            
            if len(grads) < 2:
                # Not enough samples yet
                variance[name] = torch.zeros_like(grads[0])
            else:
                # Stack into tensor [window_size, *param_shape]
                grad_tensor = torch.stack(grads)
                
                # Variance along batch dimension (dim=0)
                variance[name] = torch.var(grad_tensor, dim=0)
        
        return variance

# Usage:
# tracker = GradientVarianceTracker(model, window_size=50)
# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         logits = model(images)
#         loss = nn.functional.cross_entropy(logits, labels)
#         loss.backward()
#         
#         tracker.step()  # Record gradient
#         
#         optimizer.step()
#         optimizer.zero_grad()
# 
# var = tracker.get_variance()
# print(var['conv1.weight'].mean())  # ~0.005 (small variance)
```

## Example 3: Computing Loss Spike via Taylor

```python
def compute_loss_spike_taylor(model, train_loader, device='cuda', num_batches=50):
    """
    Compute loss spike approximation using Taylor expansion.
    
    Taylor approx: δL_i ≈ (grad_i * weight_i)^2
    
    This is a first-order approximation of how much loss increases if we remove weight_i.
    """
    
    model.eval()
    loss_spike = {}
    for name, param in model.named_parameters():
        loss_spike[name] = torch.zeros_like(param)
    
    batch_count = 0
    
    for images, labels in train_loader:
        if batch_count >= num_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Backward
        loss.backward()
        
        # Compute Taylor criterion
        for name, param in model.named_parameters():
            if param.grad is not None:
                taylor_approx = (param.grad * param) ** 2
                loss_spike[name] += taylor_approx
        
        model.zero_grad()
        batch_count += 1
    
    # Average
    for name in loss_spike:
        loss_spike[name] /= batch_count
    
    return loss_spike

# Why squared?
# Linear approximation: δL ≈ grad * δw
# Setting w → 0: δw = -w
# δL ≈ grad * (-w) = -grad*w
# But we care about magnitude, not sign, so square it: (grad*w)^2
```

## Example 4: Putting It Together — Composite Forgetting Score

```python
def compute_forgetting_score(model, train_loader, device='cuda',
                            alpha=0.5, beta=0.3, gamma=0.2):
    """
    Compute the composite Forgetting Score.
    
    S_i = α * F̂_i + β * σ²(g_i) + γ * δL_i
    """
    
    # Collect all three signals
    fisher = compute_fisher(model, train_loader, device=device, num_batches=30)
    
    # For gradient variance, we need to run training for a few steps
    tracker = GradientVarianceTracker(model, window_size=20)
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 20:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        tracker.step()
        # (skip optimizer.step for this demo)
    
    grad_variance = tracker.get_variance()
    loss_spike = compute_loss_spike_taylor(model, train_loader, device=device, num_batches=30)
    
    # Normalize each signal to [0, 1]
    fisher_norm = {}
    grad_var_norm = {}
    loss_spike_norm = {}
    
    for name in fisher:
        f_min, f_max = fisher[name].min(), fisher[name].max()
        fisher_norm[name] = (fisher[name] - f_min) / (f_max - f_min + 1e-8)
        
        g_min, g_max = grad_variance[name].min(), grad_variance[name].max()
        grad_var_norm[name] = (grad_variance[name] - g_min) / (g_max - g_min + 1e-8)
        
        l_min, l_max = loss_spike[name].min(), loss_spike[name].max()
        loss_spike_norm[name] = (loss_spike[name] - l_min) / (l_max - l_min + 1e-8)
    
    # Composite
    forgetting_score = {}
    for name in fisher:
        forgetting_score[name] = (
            alpha * fisher_norm[name] +
            beta * grad_var_norm[name] +
            gamma * loss_spike_norm[name]
        )
    
    return forgetting_score

# Usage:
# model = ResNet56(num_classes=10).to('cuda')
# train_loader = DataLoader(train_dataset, batch_size=128)
# fs = compute_forgetting_score(model, train_loader)
# print(fs['conv1.weight'].mean())  # ~0.38 (normalized)
```

---

# DEBUGGING AND COMMON PITFALLS

## Pitfall 1: Confusing Gradient Direction

**Mistake:**
```python
# Thinking: large gradient = important
fisher[name] += param.grad  # WRONG: just accumulating gradients
```

**Correct:**
```python
# Gradient magnitude squared = Fisher Information
fisher[name] += param.grad ** 2  # RIGHT: squared gradients
```

**Why:** Gradient can be positive or negative. Squaring captures the magnitude.

---

## Pitfall 2: Forgetting to Normalize Signals

**Mistake:**
```python
# Different signals have different scales
fisher_vals = [0.001, 0.005, 0.002]          # tiny values
grad_var_vals = [1000, 500, 800]            # huge values
loss_spike_vals = [0.0001, 0.00002, 0.00005]  # tiny values

# Combining directly makes gradient variance dominate
forgetting_score = fisher_vals + grad_var_vals + loss_spike_vals  # WRONG
```

**Correct:**
```python
# Normalize each to [0, 1] first
fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min())
grad_var_norm = (grad_var - grad_var.min()) / (grad_var.max() - grad_var.min())
loss_spike_norm = (loss_spike - loss_spike.min()) / (loss_spike.max() - loss_spike.min())

# Now combine
forgetting_score = 0.5*fisher_norm + 0.3*grad_var_norm + 0.2*loss_spike_norm  # RIGHT
```

**Why:** Different signal magnitudes would cause one to dominate the composite score.

---

## Pitfall 3: Using model.eval() vs model.train()

**Mistake:**
```python
model.eval()  # Freeze batch norm, disable dropout
# Compute Fisher, apply gradients, but batch norm statistics are fixed
# This gives inaccurate gradient estimates
```

**Correct:**
```python
model.train()  # Batch norm updates, dropout active
# Compute Fisher
# This gives gradients as they would be during actual training
```

**Why:** Batch norm and dropout behavior differ in eval() vs train(). To get accurate importance estimates, use the same mode as actual training.

---

## Pitfall 4: Mask Not Re-applied in Fine-tuning

**Mistake:**
```python
for step in range(n_steps):
    loss.backward()
    optimizer.step()
    # WRONG: zeroed weights are now non-zero again, contributing to forward pass
```

**Correct:**
```python
for step in range(n_steps):
    loss.backward()
    optimizer.step()
    # Re-apply mask so pruned weights = 0 again
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            module.weight.data *= module.weight_mask  # RIGHT
```

**Why:** SGD updates all weights. Pruned (zeroed) weights would gradually grow non-zero, "un-pruning" them.

---

## Pitfall 5: Using Final Checkpoint Instead of Best

**Mistake:**
```python
model = ResNet56()
model.load_state_dict(torch.load('resnet56_last.pth'))  # epoch 200, 92.90% acc
# Start pruning from a weaker model
```

**Correct:**
```python
model = ResNet56()
model.load_state_dict(torch.load('resnet56_best.pth'))  # epoch 107, 93.24% acc
# Start pruning from the strongest model
```

**Why:** Fair comparison requires giving all methods the same starting point — the best validation accuracy.

---

## Pitfall 6: Not Checking for NaN / Inf in Scores

**Mistake:**
```python
forgetting_score = fisher + grad_var + loss_spike
# If any signal has NaN (e.g., loss_spike division by zero), contamination spreads
pruned = forgetting_score < threshold  # NaN < X is False, mask computed wrongly
```

**Correct:**
```python
# Normalize with epsilon to avoid division by zero
fisher_norm = (fisher - fisher.min()) / (fisher.max() - fisher.min() + 1e-8)

# Check for NaN
assert not torch.isnan(forgetting_score).any(), "NaN in forgetting score!"

pruned = forgetting_score < threshold  # Safe
```

**Why:** Numerical instability can silently corrupt scores, leading to wrong pruning decisions.

---

## Pitfall 7: Very Small Learning Rate in Fine-tuning

**Mistake:**
```python
# Fine-tuning with initial learning rate
optimizer = SGD(model.parameters(), lr=0.1)
# The model will diverge or training will be very slow
```

**Correct:**
```python
# Fine-tuning with much lower learning rate
optimizer = SGD(model.parameters(), lr=0.005)
# Low rate: surviving weights adjust minimally, capacity redistributes
```

**Why:** Initial training learns from scratch. Fine-tuning starts from a trained model. A high learning rate overwrites existing knowledge.

---

## Debugging Checklist

Before running a big experiment:

- [ ] Verify Fisher values are non-negative (always true for squared gradients)
- [ ] Verify gradient variance is non-negative (always true for variance)
- [ ] Verify loss_spike values are non-negative (always true for squared quantities)
- [ ] Check that normalized signals are in [0, 1]
- [ ] Check that forgetting_score values are in [0, 1] (after normalization)
- [ ] Verify masks are actually zero (print model.weight.data after applying mask)
- [ ] Run quick mode (--quick flag) before full experiment to catch errors early
- [ ] Check that recovery slope is reasonable (0 to 2 %/step for healthy pruning)
- [ ] Verify no NaN/Inf in any computed quantities

---

**End of LEARNING_GUIDE_PHASES_1_3.md**

*Use this document to understand the theory, check the code examples, and debug common issues.*
*When you run the experiments, compare your results against the expected ranges and investigate any anomalies.*
