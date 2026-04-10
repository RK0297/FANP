"""
Component Ablation Study for FANP.

Tests FANP at a fixed target sparsity with each importance component
systematically disabled.  Isolates the contribution of each signal.

Ablation configurations
-----------------------
  A. FANP-Full    alpha=0.5, beta=0.3, gamma=0.2  (default)
  B. Fisher-only  alpha=1.0, beta=0.0, gamma=0.0
  C. GradVar-only alpha=0.0, beta=1.0, gamma=0.0
  D. Taylor-only  alpha=0.0, beta=0.0, gamma=1.0
  E. No-Adaptive  Full scores but tau=inf so the rate never halves
  F. Magnitude    Han et al. baseline (no FANP signals)

Every configuration uses the same pretrained checkpoint, the same random
seed, and the same recovery fine-tuning budget — so accuracy differences
are entirely due to scoring, not fine-tuning budget.

Running
-------
Quick check (minutes):
    python experiments/ablations/component_ablation.py --quick

Full study (hours — use for paper results):
    python experiments/ablations/component_ablation.py
"""
from __future__ import annotations

import sys
import os
import json
import argparse
import platform
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn

from data.cifar import get_cifar10_loaders
from models.resnet import resnet56
from experiments.baselines.magnitude import MagnitudePruner
from pruning.engine.adaptive_scheduler import AdaptivePruningScheduler
from pruning.recovery.tracker import RecoveryTracker
from metrics.sparsity import global_sparsity


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _resolve_project_path(path: str | None) -> str | None:
    """Resolve relative paths from fanp project root."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _append_run_id(path: str, run_id: str, ext: str = ".json") -> str:
    """Append run_id before extension to avoid overwrite across runs."""
    root, old_ext = os.path.splitext(path)
    final_ext = old_ext or ext
    if root.endswith(f"_{run_id}"):
        return root + final_ext
    return f"{root}_{run_id}{final_ext}"


def collect_repro_metadata(seed: int | None = None) -> dict:
    """Capture reproducibility metadata for ablation outputs."""
    metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seed": seed,
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"
    metadata["git_commit"] = commit
    return metadata


# ---------------------------------------------------------------------------
# Ablation configuration table
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: list[dict] = [
    {
        "name":  "FANP-Full",
        "alpha": 0.5, "beta": 0.3, "gamma": 0.2,
        "tau":   0.5,
        "desc":  "Full FANP (Fisher + GradVar + Taylor, adaptive rate)",
    },
    {
        "name":  "Fisher-only",
        "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
        "tau":   0.5,
        "desc":  "Fisher only — GradVar and Taylor disabled",
    },
    {
        "name":  "GradVar-only",
        "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
        "tau":   0.5,
        "desc":  "Gradient Variance only — Fisher and Taylor disabled",
    },
    {
        "name":  "Taylor-only",
        "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
        "tau":   0.5,
        "desc":  "Taylor criterion only — Fisher and GradVar disabled",
    },
    {
        # tau=2.0 is always > mean_FS (which lives in [0,1]) so the rate
        # never halves — effectively a fixed-rate scheduler.
        "name":  "No-Adaptive",
        "alpha": 0.5, "beta": 0.3, "gamma": 0.2,
        "tau":   2.0,
        "desc":  "Full FANP scores but fixed pruning rate (no adaptive slow-down)",
    },
]

DEFAULT_ABL_CFG: dict = {
    "target_sparsity":  0.70,
    "checkpoint_path":  "./checkpoints/resnet56_best.pth",
    "output_path":      "./results/ablation.json",

    "data_dir":    "./data/downloads",
    "batch_size":  128,
    "num_workers": 0,

    "base_rate":   0.10,
    "max_rounds":  30,
    "acc_batches": 20,

    "ft_n_steps":       500,
    "ft_lr":            0.005,
    "ft_eval_interval": 100,

    "device": "cuda",
    "seed":   42,

    # Quick mode overrides
    "quick_mode":            False,
    "quick_acc_batches":     3,
    "quick_ft_steps":        20,
    "quick_ft_eval_interval": 10,
    "quick_max_rounds":       3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pretrained(path: str, device: torch.device) -> nn.Module:
    model = resnet56(num_classes=10).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def count_params(model: nn.Module) -> dict:
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return {"total": total, "nonzero": nonzero}


@torch.no_grad()
def eval_acc(model, loader, criterion, device) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        correct += model(inputs).argmax(1).eq(targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Run one ablation configuration
# ---------------------------------------------------------------------------

def run_one_ablation(
    abl:          dict,
    cfg:          dict,
    train_loader,
    val_loader,
    test_loader,
    criterion:    nn.Module,
    device:       torch.device,
) -> dict:
    """Run a single FANP ablation configuration and return accuracy."""
    print(f"\n--- [{abl['name']}] {abl['desc']} ---")

    model = load_pretrained(cfg["checkpoint_path"], device)
    model.register_gradient_hooks()

    scheduler = AdaptivePruningScheduler(
        model       = model,
        criterion   = criterion,
        base_rate   = cfg["base_rate"],
        tau         = abl["tau"],
        alpha       = abl["alpha"],
        beta        = abl["beta"],
        gamma       = abl["gamma"],
        acc_batches = cfg["acc_batches"],
        device      = device,
    )

    tracker = RecoveryTracker(
        model         = model,
        masks         = scheduler.masks,
        criterion     = criterion,
        device        = device,
        n_steps       = cfg["ft_n_steps"],
        lr            = cfg["ft_lr"],
        eval_interval = cfg["ft_eval_interval"],
    )

    abl_round_log: list = []
    round_idx_counter = [0]

    def recovery_fn(m):
        sp  = global_sparsity(m)
        idx = round_idx_counter[0]
        met = tracker.measure(
            train_loader, val_loader,
            round_idx=idx, sparsity=sp,
            verbose=False,       # keep ablation output compact
        )
        step_acc_ser = [[s, a] for s, a in met.trace.step_acc] if met.trace else []
        abl_round_log.append({
            "round":         idx,
            "sparsity":      sp,
            "val_before_ft": met.acc_before,
            "val_after_ft":  met.acc_after,
            "val_ft_gain":   met.acc_after - met.acc_before,
            "slope":         met.recovery_slope,
            "overpruned":    met.overpruned,
            "step_acc":      step_acc_ser,
        })
        round_idx_counter[0] += 1

    history = scheduler.prune_to_target(
        loader          = train_loader,
        target_sparsity = cfg["target_sparsity"],
        max_rounds      = cfg["max_rounds"],
        recovery_fn     = recovery_fn,
        verbose         = True,
    )

    # Merge scheduler per-round history into abl_round_log
    for i, h in enumerate(history):
        if i < len(abl_round_log):
            abl_round_log[i]["mean_fs"]   = h.get("mean_fs",   None)
            abl_round_log[i]["rate_used"] = h.get("rate_used", None)
            abl_round_log[i]["n_pruned"]  = h.get("n_pruned",  None)

    model.remove_gradient_hooks()
    params_after = count_params(model)
    sp  = global_sparsity(model)
    acc = eval_acc(model, test_loader, criterion, device)
    print(f"  [{abl['name']}]  sparsity={sp:.1%}  test_acc={acc:.2f}%")

    return {
        "name":           abl["name"],
        "desc":           abl["desc"],
        "alpha":          abl["alpha"],
        "beta":           abl["beta"],
        "gamma":          abl["gamma"],
        "tau":            abl["tau"],
        "sparsity":       sp,
        "test_acc":       acc,
        "pruning_rounds": len(history),
        "round_log":      abl_round_log,
        "param_total":    params_after["total"],
        "param_nonzero":  params_after["nonzero"],
    }


# ---------------------------------------------------------------------------
# Magnitude baseline
# ---------------------------------------------------------------------------

def run_magnitude_baseline(cfg, test_loader, criterion, device) -> dict:
    print(f"\n--- [Magnitude] Han et al. 2015 ---")
    model  = load_pretrained(cfg["checkpoint_path"], device)
    pruner = MagnitudePruner(model)
    pruner.prune(sparsity=cfg["target_sparsity"])
    sp  = global_sparsity(model)
    acc = eval_acc(model, test_loader, criterion, device)
    print(f"  [Magnitude]  sparsity={sp:.1%}  test_acc={acc:.2f}%")
    return {"name": "Magnitude", "desc": "Baseline (Han et al. 2015)", "sparsity": sp, "test_acc": acc}


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_ablation_table(results: list, target_sparsity: float) -> None:
    baseline_acc = results[0]["test_acc"]   # Magnitude is always first

    print(f"\n{'='*58}")
    print(f"  ABLATION @ ~{target_sparsity:.0%} sparsity")
    print(f"{'='*58}")
    print(f"{'Method':<20}  {'Sparsity':>9}  {'Test Acc':>9}  {'vs Magnitude':>13}")
    print("-" * 58)
    for r in results:
        delta = r["test_acc"] - baseline_acc
        delta_str = f"({delta:+.2f}%)"
        print(f"{r['name']:<20}  {r['sparsity']*100:>8.1f}%  "
              f"{r['test_acc']:>8.2f}%  {delta_str:>13}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Top-level ablation runner
# ---------------------------------------------------------------------------

def run_ablation(cfg: dict | None = None) -> list:
    """
    Run all ablation configurations and return results.

    Parameters
    ----------
    cfg : dict, optional
        Configuration overrides.  Falls back to DEFAULT_ABL_CFG for missing keys.

    Returns
    -------
    list[dict]
        One result dict per ablation configuration.
    """
    if cfg is None:
        cfg = DEFAULT_ABL_CFG.copy()
    else:
        merged = DEFAULT_ABL_CFG.copy()
        merged.update(cfg)
        cfg = merged

    if cfg.get("quick_mode", False):
        print("[quick_mode] Overriding to fast settings for testing.")
        cfg = cfg.copy()
        cfg["acc_batches"]    = cfg["quick_acc_batches"]
        cfg["ft_n_steps"]     = cfg["quick_ft_steps"]
        cfg["ft_eval_interval"] = cfg["quick_ft_eval_interval"]
        cfg["max_rounds"]     = cfg["quick_max_rounds"]

    cfg["checkpoint_path"] = _resolve_project_path(cfg["checkpoint_path"])
    cfg["data_dir"] = _resolve_project_path(cfg["data_dir"])
    if cfg.get("output_path"):
        cfg["output_path"] = _resolve_project_path(cfg["output_path"])

    run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["run_id"] = run_id

    torch.manual_seed(cfg["seed"])
    device    = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    print(f"\nAblation Study — target sparsity: {cfg['target_sparsity']:.0%}")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir    = cfg["data_dir"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
    )

    all_results: list = []

    # Magnitude baseline comes first (used as delta reference in the table)
    all_results.append(
        run_magnitude_baseline(cfg, test_loader, criterion, device)
    )

    # FANP ablations
    for abl in ABLATION_CONFIGS:
        result = run_one_ablation(
            abl, cfg, train_loader, val_loader, test_loader, criterion, device
        )
        all_results.append(result)

    print_ablation_table(all_results, cfg["target_sparsity"])

    # Save results
    out_path = _append_run_id(cfg["output_path"], run_id, ext=".json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "run_id": run_id,
                "run_timestamp": run_id,
                "checkpoint": cfg["checkpoint_path"],
                "config": {k: v for k, v in cfg.items() if not k.startswith("quick_")},
                "repro": collect_repro_metadata(seed=cfg.get("seed")),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"Ablation results saved to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FANP component ablation study")
    parser.add_argument("--quick", action="store_true",
                        help="Run in quick mode for fast testing")
    parser.add_argument("--sparsity", type=float, default=None,
                        help="Target sparsity (default: 0.70)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: results/ablation_<timestamp>.json)")
    args = parser.parse_args()

    from datetime import datetime
    cfg = DEFAULT_ABL_CFG.copy()
    if args.quick:
        cfg["quick_mode"] = True
    if args.sparsity is not None:
        cfg["target_sparsity"] = args.sparsity
    if args.out is not None:
        cfg["output_path"] = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["output_path"] = f"./results/ablation_{ts}.json"

    run_ablation(cfg)
