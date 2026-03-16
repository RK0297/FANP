"""
Phase 3 smoke tests.

Runs every Phase 3 component on tiny dummy data to catch import errors,
API mismatches, and shape bugs — before committing to a long GPU run.

All tests use:
  - 160 dummy samples, batch_size=32
  - 5 fine-tuning steps
  - 2 scoring batches
  - A freshly randomly-initialised ResNet-56 (no pretrained weights)

Expected output: ALL PHASE 3 TESTS PASSED
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from models.resnet import resnet56

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES  = 160
BATCH_SIZE = 32

def make_loader(n: int = N_SAMPLES, bs: int = BATCH_SIZE):
    x  = torch.randn(n, 3, 32, 32)
    y  = torch.randint(0, 10, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

def make_model():
    m = resnet56(num_classes=10).to(DEVICE)
    m.register_gradient_hooks()
    return m

CRITERION = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Mini test harness
# ---------------------------------------------------------------------------

PASSES: int = 0
FAILS:  list = []

def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        print(f"  PASS  {name}")
        PASSES += 1
    else:
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)
        FAILS.append(name)


# ===========================================================================
# Test 1: FineTuner
# ===========================================================================

print("\n[1] FineTuner ...")
try:
    from pruning.recovery.fine_tuner import FineTuner, RecoveryTrace

    model  = make_model()
    loader = make_loader()

    # All-ones masks — nothing is pruned; fine-tuner should still run cleanly
    masks = {
        name: torch.ones_like(param.data)
        for name, param in model.named_parameters()
        if "conv" in name and "weight" in name
    }

    ft    = FineTuner(model=model, masks=masks, criterion=CRITERION, device=DEVICE)
    trace = ft.fine_tune(
        train_loader=loader,
        val_loader=loader,
        n_steps=5,
        lr=0.01,
        eval_interval=5,
        verbose=True,
    )

    check("FineTuner returns RecoveryTrace",   isinstance(trace, RecoveryTrace))
    check("step_acc has at least 2 entries",   len(trace.step_acc) >= 2)
    check("initial_acc is a float",            isinstance(trace.initial_acc, float))
    check("final_acc is a float",              isinstance(trace.final_acc,   float))
    check("recovery_slope is finite",
          not (trace.recovery_slope != trace.recovery_slope))   # NaN check

except Exception as exc:
    print(f"  ERROR  FineTuner raised {type(exc).__name__}: {exc}")
    FAILS.append("FineTuner")


# ===========================================================================
# Test 2: RecoveryTracker
# ===========================================================================

print("\n[2] RecoveryTracker ...")
try:
    from pruning.recovery.tracker import RecoveryTracker, RecoveryMetrics

    model  = make_model()
    loader = make_loader()
    masks  = {
        name: torch.ones_like(param.data)
        for name, param in model.named_parameters()
        if "conv" in name and "weight" in name
    }

    tracker = RecoveryTracker(
        model=model, masks=masks, criterion=CRITERION, device=DEVICE,
        n_steps=5, lr=0.01, eval_interval=5,
        slope_threshold=0.0,    # accept any slope (random init)
    )
    metrics = tracker.measure(loader, loader, round_idx=0, sparsity=0.30, verbose=True)

    check("RecoveryTracker returns RecoveryMetrics", isinstance(metrics, RecoveryMetrics))
    check("acc_before is a float",                   isinstance(metrics.acc_before, float))
    check("acc_after  is a float",                   isinstance(metrics.acc_after,  float))
    check("overpruned flag exists",                  hasattr(metrics, "overpruned"))
    check("history length == 1",                     len(tracker.history) == 1)

    tracker.summary()

except Exception as exc:
    print(f"  ERROR  RecoveryTracker raised {type(exc).__name__}: {exc}")
    FAILS.append("RecoveryTracker")


# ===========================================================================
# Test 3: StructuredFANPPruner  (needs torch_pruning)
# ===========================================================================

print("\n[3] StructuredFANPPruner ...")
try:
    import torch_pruning                    # noqa: F401 — check importable
    from pruning.engine.structured import StructuredFANPPruner

    model  = make_model()
    loader = make_loader(n=64, bs=32)

    pruner = StructuredFANPPruner(
        model=model, criterion=CRITERION, device=DEVICE,
        acc_batches=2, window_K=5,
    )
    stats = pruner.prune(loader=loader, pruning_ratio=0.20)

    check("prune() returns a dict",            isinstance(stats, dict))
    check("params_after < params_before",      stats["params_after"] < stats["params_before"])
    check("compression > 1.0",                 stats["compression"] > 1.0)

except ImportError as exc:
    print(f"  SKIP  StructuredFANPPruner (torch_pruning not available: {exc})")
except Exception as exc:
    print(f"  ERROR  StructuredFANPPruner raised {type(exc).__name__}: {exc}")
    FAILS.append("StructuredFANPPruner")


# ===========================================================================
# Test 4: main_experiment — quick mode, single sparsity
# ===========================================================================

print("\n[4] main_experiment (quick_mode, sparsity=0.30) ...")
try:
    from experiments.main_experiment import run_experiment, DEFAULT_CFG

    quick_cfg = {
        **DEFAULT_CFG,
        "sparsity_levels":     [0.30],
        "quick_mode":          True,
        "quick_acc_batches":   2,
        "quick_ft_steps":      5,
        "quick_ft_eval_interval": 5,
        "quick_max_rounds":    2,
        "output_path":         "./results/test_main_experiment.json",
    }
    results = run_experiment(quick_cfg)

    fanp_result  = list(results["fanp"].values())[0]
    mag_result   = list(results["magnitude"].values())[0]

    check("run_experiment returns dict",         isinstance(results, dict))
    check("results has 3 methods",               len(results) >= 3)
    check("FANP result has test_acc (float)",    isinstance(fanp_result["test_acc"], float))
    check("Magnitude result has test_acc",       isinstance(mag_result["test_acc"],  float))

except Exception as exc:
    print(f"  ERROR  main_experiment raised {type(exc).__name__}: {exc}")
    import traceback; traceback.print_exc()
    FAILS.append("main_experiment")


# ===========================================================================
# Test 5: component_ablation — quick mode, first 2 configs only
# ===========================================================================

print("\n[5] component_ablation (quick_mode, 2 configs) ...")
try:
    import experiments.ablations.component_ablation as abl_mod
    from experiments.ablations.component_ablation import run_ablation, DEFAULT_ABL_CFG

    quick_abl_cfg = {
        **DEFAULT_ABL_CFG,
        "quick_mode":            True,
        "quick_acc_batches":     2,
        "quick_ft_steps":        5,
        "quick_ft_eval_interval": 5,
        "quick_max_rounds":      2,
        "output_path":           "./results/test_ablation.json",
    }

    # Temporarily run only the first 2 ablation configs to save time
    original_configs          = abl_mod.ABLATION_CONFIGS
    abl_mod.ABLATION_CONFIGS  = original_configs[:2]

    abl_results = run_ablation(quick_abl_cfg)

    abl_mod.ABLATION_CONFIGS = original_configs    # restore

    check("run_ablation returns a list",         isinstance(abl_results, list))
    check("at least 3 results (mag + 2 abl)",    len(abl_results) >= 3)
    check("each result has test_acc",
          all("test_acc" in r for r in abl_results))

except Exception as exc:
    print(f"  ERROR  component_ablation raised {type(exc).__name__}: {exc}")
    import traceback; traceback.print_exc()
    FAILS.append("component_ablation")


# ===========================================================================
# Summary
# ===========================================================================

total = PASSES + len(FAILS)
print(f"\n{'='*52}")
print(f"Phase 3 tests: {PASSES}/{total} passed")
if FAILS:
    print(f"Failed tests : {FAILS}")
else:
    print("ALL PHASE 3 TESTS PASSED")
print("=" * 52)
