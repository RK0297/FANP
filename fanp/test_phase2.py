"""
Phase 2 smoke test — verifies all importance estimators and the adaptive scheduler.

Run from fanp/ directory:
    python test_phase2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Load trained model ───────────────────────────────────────────────────────
from models.resnet import resnet56
from pruning.importance.fisher import EmpiricalFisher
from pruning.importance.gradient_variance import GradientVariance
from pruning.importance.loss_spike import LossSpike
from pruning.importance.composite import ForgettingScore
from pruning.engine.adaptive_scheduler import AdaptivePruningScheduler
from metrics.sparsity import global_sparsity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = "./checkpoints/resnet56_best.pth"

print(f"Device: {DEVICE}")
print("=" * 60)

# ── Build dummy mini-loader (5 batches, 32 samples each) ─────────────────────
torch.manual_seed(0)
dummy_x = torch.randn(160, 3, 32, 32)
dummy_y = torch.randint(0, 10, (160,))
mini_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=32, shuffle=False)
criterion   = nn.CrossEntropyLoss()

# ── Load checkpoint ──────────────────────────────────────────────────────────
model = resnet56(num_classes=10).to(DEVICE)
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: val_acc = {ckpt['val_acc']:.2f}%")
else:
    print("WARNING: checkpoint not found — using random weights")

# ── Test 1: EmpiricalFisher ──────────────────────────────────────────────────
print("\n[1] EmpiricalFisher")
fisher = EmpiricalFisher(model, device=DEVICE)
for inputs, targets in mini_loader:
    fisher.accumulate(model, inputs, targets, criterion)
f_scores = fisher.scores()
print(f"    Layers scored: {len(f_scores)}")
sample_name = next(iter(f_scores))
print(f"    Sample layer: {sample_name}  shape={f_scores[sample_name].shape}  mean={f_scores[sample_name].mean():.6f}")
fisher.reset()
print("    PASS")

# ── Test 2: GradientVariance ─────────────────────────────────────────────────
print("\n[2] GradientVariance")
gv = GradientVariance(model, window_K=5, device=DEVICE)
for inputs, targets in mini_loader:
    gv.accumulate(model, inputs, targets, criterion)
gv_scores = gv.scores()
print(f"    Layers scored: {len(gv_scores)}")
print(f"    Sample layer: {sample_name}  mean variance={gv_scores[sample_name].mean():.6f}")
gv.reset()
print("    PASS")

# ── Test 3: LossSpike ────────────────────────────────────────────────────────
print("\n[3] LossSpike (fast layer-wise approximation)")
ls = LossSpike(model, device=DEVICE, n_batches=3)
for inputs, targets in mini_loader:
    ls.accumulate(model, inputs, targets, criterion)
ls_scores = ls.scores(criterion=criterion)
print(f"    Layers scored: {len(ls_scores)}")
print(f"    Sample layer: {sample_name}  mean delta_L={ls_scores[sample_name].mean():.6f}")
ls.reset()
print("    PASS")

# ── Test 4: ForgettingScore (composite) ──────────────────────────────────────
print("\n[4] ForgettingScore (composite: Fisher + GradVar + LossSpike)")
fs_engine = ForgettingScore(model, alpha=0.5, beta=0.3, gamma=0.2,
                             window_K=5, n_batches=3, device=DEVICE)
for inputs, targets in mini_loader:
    fs_engine.accumulate(inputs, targets, criterion)
composite = fs_engine.compute(criterion)
print(f"    Layers scored: {len(composite)}")
fs_engine.summary(composite)
fs_engine.reset()
print("    PASS")

# ── Test 5: AdaptivePruningScheduler ─────────────────────────────────────────
print("\n[5] AdaptivePruningScheduler (2 rounds to 20% sparsity)")
import copy
model_copy = copy.deepcopy(model).to(DEVICE)
scheduler = AdaptivePruningScheduler(
    model_copy, criterion,
    base_rate=0.10, tau=0.5,
    alpha=0.5, beta=0.3, gamma=0.2,
    window_K=5, acc_batches=3,
    device=DEVICE,
)
print(f"    Initial sparsity: {global_sparsity(model_copy):.3%}")
history = scheduler.prune_to_target(
    mini_loader, target_sparsity=0.20, max_rounds=5, verbose=True
)
print(f"    Final sparsity:  {global_sparsity(model_copy):.3%}")
print(f"    Rounds completed: {len(history)}")
print("    PASS")

print("\n" + "=" * 60)
print("ALL PHASE 2 TESTS PASSED")
