"""
Main FANP vs Magnitude Experiment.

Compares three pruning strategies across multiple sparsity levels:

  1. Magnitude       -- Han et al. 2015, no fine-tuning  (industry baseline)
  2. Magnitude + FT  -- Same as above with recovery fine-tuning
                        (control: isolates FT contribution from smarter scoring)
  3. FANP            -- Full FANP adaptive pruner with per-round fine-tuning

Expected outcome: FANP maintains higher accuracy at high sparsity (70-90 %)
where magnitude pruning collapses.

Running
-------
  Quick sanity check (minutes):
      python experiments/main_experiment.py --quick

  Full experiment (hours on GPU):
      python experiments/main_experiment.py

  Resume an interrupted run:
      python experiments/main_experiment.py --resume results/main_20260312_143022.json

  Single sparsity level:
      python experiments/main_experiment.py --sparsity 0.70

Saving
------
  Results are saved to ./results/main_<timestamp>.json  (never overwritten).
  The same file is updated after EACH sparsity level — so if the run is
  interrupted, completed levels are preserved.  Use --resume to continue.
  A human-readable summary .txt is also written at the end.
"""
from __future__ import annotations

import sys
import os
import json
import argparse
import math
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from data.cifar import get_cifar10_loaders
from models.resnet import resnet56
from experiments.baselines.magnitude import MagnitudePruner
from pruning.engine.adaptive_scheduler import AdaptivePruningScheduler
from pruning.recovery.tracker import RecoveryTracker
from metrics.sparsity import global_sparsity


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_project_path(path: str | None) -> str | None:
    """Resolve relative paths from the fanp project root, not current cwd."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    # Which sparsity levels to test
    "sparsity_levels": [0.30, 0.50, 0.70, 0.90],

    # Path to the pretrained ResNet-56 checkpoint.
    # resnet56_best.pth = highest val_acc epoch (epoch 107, 93.24%).
    # This is intentional — we always start pruning from the best weights.
    "checkpoint_path": "./checkpoints/resnet56_best.pth",

    # Data loading
    "data_dir":    "./data/downloads",
    "batch_size":  128,
    "num_workers": 0,       # 0 required on Windows (shared memory limitation)

    # FANP adaptive scheduler
    "fanp_base_rate":   0.10,   # fraction of live weights pruned per round
    "fanp_tau":         0.50,   # mean_FS threshold — above this, rate halves
    "fanp_max_rounds":  30,     # safety cap on pruning rounds
    "fanp_acc_batches": 20,     # mini-batches used to compute scores per round

    # Recovery fine-tuning (after each FANP pruning round)
    "ft_n_steps":       500,    # SGD steps per recovery session
    "ft_lr":            0.005,  # cosine-annealed from this value to 0
    "ft_eval_interval": 100,    # steps between val_acc snapshots

    # Device / reproducibility
    "device": "cuda",
    "seed":   42,

    # Quick mode overrides (for fast testing — minutes not hours)
    "quick_mode":             False,
    "quick_acc_batches":      3,
    "quick_ft_steps":         20,
    "quick_ft_eval_interval": 10,
    "quick_max_rounds":       3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pretrained(path: str, device: torch.device) -> nn.Module:
    """
    Load a pretrained ResNet-56 from a checkpoint file.
    Always loads resnet56_best.pth = the epoch with highest val_acc
    (epoch 107, 93.24%) — not the final epoch.  Starting from the best
    weights gives a higher ceiling for the pruning experiment.
    """
    model = resnet56(num_classes=10).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch   = ckpt["epoch"]
    val_acc = ckpt["val_acc"]
    print(f"  checkpoint: {os.path.basename(path)} "
          f"| epoch {epoch} | val_acc {val_acc:.2f}% "
          f"(best checkpoint — not the final epoch)")
    return model


@torch.no_grad()
def eval_test_acc(model, loader, criterion, device) -> float:
    """Top-1 accuracy on ``loader`` (returns percent, e.g. 92.5)."""
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        correct += model(inputs).argmax(1).eq(targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / max(total, 1)


def section(title: str, width: int = 62) -> None:
    """Print a clearly visible section banner."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subsection(title: str) -> None:
    """Print a smaller method-level banner."""
    print(f"\n  --- {title} ---")


def count_params(model: nn.Module) -> dict:
    """Return total and non-zero parameter counts."""
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return {"total": total, "nonzero": nonzero}


# ---------------------------------------------------------------------------
# Strategy 1: Magnitude pruning — no fine-tuning
# ---------------------------------------------------------------------------

def run_magnitude(cfg, sparsity, test_loader, criterion, device) -> dict:
    """Prune by weight magnitude at ``sparsity``, no fine-tuning."""
    subsection(f"MAGNITUDE  (no fine-tuning)")
    model        = load_pretrained(cfg["checkpoint_path"], device)
    params_dense = count_params(model)
    t0           = time.time()
    pruner       = MagnitudePruner(model)
    pruner.prune(sparsity=sparsity)
    actual_sp     = global_sparsity(model)
    acc           = eval_test_acc(model, test_loader, criterion, device)
    wall_sec      = time.time() - t0
    params_pruned = count_params(model)
    compression   = params_dense["nonzero"] / max(params_pruned["nonzero"], 1)
    print(f"  [Magnitude]  sparsity {actual_sp:.1%}   test_acc {acc:.2f}%  ({wall_sec:.0f}s)")
    return {
        "method":            "magnitude",
        "target":            sparsity,
        "sparsity":          actual_sp,
        "test_acc":          acc,
        "param_total":       params_pruned["total"],
        "param_nonzero":     params_pruned["nonzero"],
        "param_total_dense": params_dense["total"],
        "compression_ratio": compression,
        "wall_clock_sec":    wall_sec,
    }


# ---------------------------------------------------------------------------
# Strategy 2: Magnitude + recovery fine-tuning  (control experiment)
# ---------------------------------------------------------------------------

def run_magnitude_with_ft(
    cfg, sparsity, train_loader, val_loader, test_loader, criterion, device
) -> dict:
    """
    Magnitude prune then fine-tune (control experiment).
    Shows how much of FANP's improvement comes from fine-tuning alone
    vs. from smarter importance scoring.

    Note: slope_threshold=0.0 here because the OVERPRUNED flag is
    meaningless for one-shot pruning — the slope being small just means
    the model barely lost accuracy at this sparsity level, not that it
    was hurt by pruning.
    """
    subsection("MAGNITUDE + FINE-TUNE  (control)")
    model          = load_pretrained(cfg["checkpoint_path"], device)
    params_dense   = count_params(model)
    t0             = time.time()
    pruner         = MagnitudePruner(model)
    pruner.prune(sparsity=sparsity)

    # Build binary masks from the current (pruned) weight state
    masks: dict = {}
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            masks[name] = (param.data != 0).float().to(device)

    tracker = RecoveryTracker(
        model            = model,
        masks            = masks,
        criterion        = criterion,
        device           = device,
        n_steps          = cfg["ft_n_steps"],
        lr               = cfg["ft_lr"],
        eval_interval    = cfg["ft_eval_interval"],
        slope_threshold  = 0.0,         # flag not meaningful for one-shot prune
    )
    metrics = tracker.measure(
        train_loader, val_loader,
        round_idx=0, sparsity=sparsity,
    )

    step_acc_ser  = [[s, a] for s, a in metrics.trace.step_acc] if metrics.trace else []
    actual_sp     = global_sparsity(model)
    acc           = eval_test_acc(model, test_loader, criterion, device)
    wall_sec      = time.time() - t0
    params_pruned = count_params(model)
    compression   = params_dense["nonzero"] / max(params_pruned["nonzero"], 1)
    print(f"  [Magnitude+FT]  sparsity {actual_sp:.1%}   test_acc {acc:.2f}%  ({wall_sec:.0f}s)")
    return {
        "method":            "magnitude_ft",
        "target":            sparsity,
        "sparsity":          actual_sp,
        "test_acc":          acc,
        "val_before_ft":     metrics.acc_before,
        "val_after_ft":      metrics.acc_after,
        "val_ft_gain":       metrics.acc_after - metrics.acc_before,
        "step_acc":          step_acc_ser,
        "param_total":       params_pruned["total"],
        "param_nonzero":     params_pruned["nonzero"],
        "param_total_dense": params_dense["total"],
        "compression_ratio": compression,
        "wall_clock_sec":    wall_sec,
    }


# ---------------------------------------------------------------------------
# Strategy 3: FANP — adaptive scoring with per-round fine-tuning
# ---------------------------------------------------------------------------

def run_fanp(
    cfg, sparsity, train_loader, val_loader, test_loader, criterion, device
) -> dict:
    """FANP adaptive pruner with recovery fine-tuning after each round."""
    subsection("FANP  (our method)")
    model        = load_pretrained(cfg["checkpoint_path"], device)
    params_dense = count_params(model)
    t0           = time.time()
    model.register_gradient_hooks()

    scheduler = AdaptivePruningScheduler(
        model       = model,
        criterion   = criterion,
        base_rate   = cfg["fanp_base_rate"],
        tau         = cfg["fanp_tau"],
        acc_batches = cfg["fanp_acc_batches"],
        device      = device,
    )

    tracker = RecoveryTracker(
        model         = model,
        masks         = scheduler.masks,   # shared reference — mask updates propagate
        criterion     = criterion,
        device        = device,
        n_steps       = cfg["ft_n_steps"],
        lr            = cfg["ft_lr"],
        eval_interval = cfg["ft_eval_interval"],
    )

    round_log: list = []
    round_counter   = [0]

    def recovery_fn(m):
        sp  = global_sparsity(m)
        idx = round_counter[0]
        met = tracker.measure(
            train_loader, val_loader,
            round_idx=idx, sparsity=sp,
            verbose=True,
        )
        step_acc_ser = [[s, a] for s, a in met.trace.step_acc] if met.trace else []
        round_log.append({
            "round":         idx,
            "sparsity":      sp,
            "val_before_ft": met.acc_before,
            "val_after_ft":  met.acc_after,
            "val_ft_gain":   met.acc_after - met.acc_before,
            "slope":         met.recovery_slope,
            "overpruned":    met.overpruned,
            "step_acc":      step_acc_ser,
        })
        round_counter[0] += 1

    history = scheduler.prune_to_target(
        loader          = train_loader,
        target_sparsity = sparsity,
        max_rounds      = cfg["fanp_max_rounds"],
        recovery_fn     = recovery_fn,
        verbose         = True,
    )

    # Merge scheduler per-round history (mean_fs, rate_used, n_pruned) into round_log
    for i, h in enumerate(history):
        if i < len(round_log):
            round_log[i]["mean_fs"]   = h.get("mean_fs",   None)
            round_log[i]["rate_used"] = h.get("rate_used", None)
            round_log[i]["n_pruned"]  = h.get("n_pruned",  None)

    model.remove_gradient_hooks()
    actual_sp     = global_sparsity(model)
    acc           = eval_test_acc(model, test_loader, criterion, device)
    wall_sec      = time.time() - t0
    params_pruned = count_params(model)
    compression   = params_dense["nonzero"] / max(params_pruned["nonzero"], 1)

    # Save the pruned model for Phase 4 ONNX export
    pruned_dir      = os.path.join(os.path.dirname(os.path.abspath(cfg["checkpoint_path"])), "pruned")
    os.makedirs(pruned_dir, exist_ok=True)
    model_save_path = os.path.join(pruned_dir, f"fanp_sp{sparsity:.2f}.pth")
    torch.save({"model_state": model.state_dict(), "sparsity": actual_sp, "test_acc": acc}, model_save_path)
    print(f"  Pruned model saved to {model_save_path}")

    print(f"  [FANP]  sparsity {actual_sp:.1%}   test_acc {acc:.2f}%  ({wall_sec:.0f}s)")
    tracker.summary()

    return {
        "method":            "fanp",
        "target":            sparsity,
        "sparsity":          actual_sp,
        "test_acc":          acc,
        "pruning_rounds":    len(history),
        "round_log":         round_log,
        "param_total":       params_pruned["total"],
        "param_nonzero":     params_pruned["nonzero"],
        "param_total_dense": params_dense["total"],
        "compression_ratio": compression,
        "model_save_path":   model_save_path,
        "wall_clock_sec":    wall_sec,
    }


# ---------------------------------------------------------------------------
# Results table printers
# ---------------------------------------------------------------------------

def print_sparsity_summary(results: dict, sparsity: float) -> None:
    """
    Print a compact result table for ONE sparsity level.
    Called immediately after each level completes so you can read progress
    without waiting for the whole experiment to finish.
    """
    mag_acc  = results["magnitude"].get(sparsity, {}).get("test_acc", float("nan"))
    mft_acc  = results["magnitude_ft"].get(sparsity, {}).get("test_acc", float("nan"))
    fanp_acc = results["fanp"].get(sparsity, {}).get("test_acc", float("nan"))

    def gain(a, b):
        return "N/A" if (math.isnan(a) or math.isnan(b)) else f"{a-b:+.2f}%"

    print(f"\n  PARTIAL RESULT @ {sparsity:.0%} sparsity")
    print(f"  {'Method':<16} {'Test Acc':>10}  {'vs Magnitude':>13}")
    print(f"  {'-'*42}")
    print(f"  {'Magnitude':<16} {mag_acc:>9.2f}%  {'baseline':>13}")
    print(f"  {'Magnitude+FT':<16} {mft_acc:>9.2f}%  {gain(mft_acc, mag_acc):>13}")
    print(f"  {'FANP':<16} {fanp_acc:>9.2f}%  {gain(fanp_acc, mag_acc):>13}")


def print_final_table(results: dict, sparsity_levels: list) -> str:
    """
    Print and return the full comparison table across all sparsity levels.
    Also used for writing the .txt summary file.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("  FANP vs Magnitude Pruning — Final Results")
    lines.append("=" * 72)
    lines.append(
        f"  {'Sparsity':>8}  {'Magnitude':>10}  {'Mag+FT':>10}  "
        f"{'FANP':>10}  {'FANP-Mag':>10}  {'FANP-MagFT':>11}"
    )
    lines.append("  " + "-" * 68)
    for sp in sparsity_levels:
        mag_acc  = results["magnitude"].get(sp, {}).get("test_acc", float("nan"))
        mft_acc  = results["magnitude_ft"].get(sp, {}).get("test_acc", float("nan"))
        fanp_acc = results["fanp"].get(sp, {}).get("test_acc", float("nan"))

        def fmt(a, b):
            return "   N/A  " if (math.isnan(a) or math.isnan(b)) else f"{a-b:+.2f}%  "

        lines.append(
            f"  {sp*100:>7.0f}%  "
            f"{mag_acc:>9.2f}%  "
            f"{mft_acc:>9.2f}%  "
            f"{fanp_acc:>9.2f}%  "
            f"{fmt(fanp_acc, mag_acc):>10}  "
            f"{fmt(fanp_acc, mft_acc):>11}"
        )
    lines.append("=" * 72)
    table = "\n".join(lines)
    print("\n" + table)
    return table


# ---------------------------------------------------------------------------
# Incremental saving helpers
# ---------------------------------------------------------------------------

def _save_results(results: dict, out_path: str) -> None:
    """Write results JSON.  Called after every sparsity level."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _save_summary_txt(table: str, out_path: str) -> None:
    """Write the human-readable results table to a .txt file."""
    txt_path = out_path.replace(".json", "_summary.txt")
    with open(txt_path, "w") as f:
        f.write(table + "\n")
    print(f"Summary saved to  {txt_path}")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict | None = None) -> dict:
    """
    Run the full FANP vs Magnitude comparison experiment.

    Results are saved after EACH sparsity level (not just at the end), so
    an interrupted run can be resumed with --resume.

    Parameters
    ----------
    cfg : dict, optional
        Configuration overrides.  Missing keys fall back to DEFAULT_CFG.

    Returns
    -------
    dict
        ``{"magnitude": {sp: result}, "magnitude_ft": {sp: result}, "fanp": {sp: result}}``
    """
    if cfg is None:
        cfg = DEFAULT_CFG.copy()
    else:
        merged = DEFAULT_CFG.copy()
        merged.update(cfg)
        cfg = merged

    if cfg.get("quick_mode", False):
        print("[quick_mode] Overriding to fast settings for testing.")
        cfg = cfg.copy()
        cfg["fanp_acc_batches"] = cfg["quick_acc_batches"]
        cfg["ft_n_steps"]       = cfg["quick_ft_steps"]
        cfg["ft_eval_interval"] = cfg["quick_ft_eval_interval"]
        cfg["fanp_max_rounds"]  = cfg["quick_max_rounds"]

    cfg["checkpoint_path"] = _resolve_project_path(cfg["checkpoint_path"])
    cfg["data_dir"]        = _resolve_project_path(cfg["data_dir"])
    if cfg.get("output_path"):
        cfg["output_path"] = _resolve_project_path(cfg["output_path"])
    if cfg.get("resume_path"):
        cfg["resume_path"] = _resolve_project_path(cfg["resume_path"])

    torch.manual_seed(cfg["seed"])
    device    = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Build timestamped output path (never overwrite a previous run)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.get("output_path") or _resolve_project_path(f"./results/main_{ts}.json")
    # If not already timestamped, inject the timestamp
    if os.path.normpath(out_path) == os.path.normpath(_resolve_project_path("./results/main_experiment.json")):
        out_path = _resolve_project_path(f"./results/main_{ts}.json")

    sparsity_levels = cfg["sparsity_levels"]

    # --- Resume: load partial results from a previous run --------------------
    resume_path = cfg.get("resume_path")
    if resume_path and os.path.isfile(resume_path):
        with open(resume_path) as f:
            results = json.load(f)
        # JSON keys are strings; convert back to float (skip "meta" which has string keys)
        for method in results:
            if method == "meta":
                continue
            results[method] = {float(k): v for k, v in results[method].items()}
        out_path = resume_path   # continue writing to the same file
        completed = set(results["magnitude"].keys())
        print(f"Resuming from {resume_path}")
        print(f"Already completed sparsity levels: {[f'{s:.0%}' for s in sorted(completed)]}")
        sparsity_levels = [s for s in sparsity_levels if s not in completed]
        if not sparsity_levels:
            print("All sparsity levels already completed. Nothing to do.")
            return results
    else:
        results = {"magnitude": {}, "magnitude_ft": {}, "fanp": {}, "meta": {}}

    print(f"\nDevice:    {device}")
    print(f"Levels:    {[f'{s:.0%}' for s in sparsity_levels]}")
    print(f"Saving to: {out_path}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir    = cfg["data_dir"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
    )

    # Evaluate dense baseline once (skip if resuming and meta already present)
    if not results.get("meta"):
        section("DENSE BASELINE")
        _dense_model  = load_pretrained(cfg["checkpoint_path"], device)
        _dense_acc    = eval_test_acc(_dense_model, test_loader, criterion, device)
        _dense_params = count_params(_dense_model)
        del _dense_model
        print(f"  Dense baseline: test_acc={_dense_acc:.2f}%  params={_dense_params['total']:,}")
        results["meta"] = {
            "dense_test_acc":      _dense_acc,
            "param_total_dense":   _dense_params["total"],
            "param_nonzero_dense": _dense_params["nonzero"],
            "run_timestamp":       ts,
            "checkpoint":          cfg["checkpoint_path"],
            "config":              {k: v for k, v in cfg.items() if not k.startswith("quick_")},
        }
        _save_results(results, out_path)

    for sparsity in sparsity_levels:
        section(f"TARGET SPARSITY: {sparsity:.0%}")

        results["magnitude"][sparsity] = run_magnitude(
            cfg, sparsity, test_loader, criterion, device
        )
        results["magnitude_ft"][sparsity] = run_magnitude_with_ft(
            cfg, sparsity, train_loader, val_loader, test_loader, criterion, device
        )
        results["fanp"][sparsity] = run_fanp(
            cfg, sparsity, train_loader, val_loader, test_loader, criterion, device
        )

        # Print mini-table immediately and save — so interrupted runs are not lost
        print_sparsity_summary(results, sparsity)
        _save_results(results, out_path)

    table = print_final_table(results, cfg["sparsity_levels"])
    _save_summary_txt(table, out_path)
    print(f"Full results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FANP vs Magnitude pruning comparison experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/main_experiment.py                     # full run
  python experiments/main_experiment.py --quick             # fast test (~minutes)
  python experiments/main_experiment.py --sparsity 0.7 0.9  # specific levels
  python experiments/main_experiment.py --resume results/main_20260312_143022.json
        """,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run in quick mode (tiny batches/steps for fast testing)",
    )
    parser.add_argument(
        "--sparsity", nargs="+", type=float,
        help="Sparsity levels to test, e.g. --sparsity 0.5 0.7",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help="Path to a partial results JSON to resume an interrupted run",
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="PATH",
        help="Override the output JSON path (default: results/main_<timestamp>.json)",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()

    if args.quick:
        cfg["quick_mode"] = True
    if args.sparsity:
        cfg["sparsity_levels"] = args.sparsity
    if args.resume:
        cfg["resume_path"] = args.resume
    if args.out:
        cfg["output_path"] = args.out

    run_experiment(cfg)
