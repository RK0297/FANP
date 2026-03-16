"""
Pruning evaluation script.

Loads a trained checkpoint → applies magnitude pruning at multiple
sparsity levels → reports accuracy + sparsity curve.

This generates the baseline curve that FANP must beat in Phase 4.

Run:
    python eval_pruning.py
    (loads checkpoints/resnet56_best.pth or uses a freshly init'd model for smoke tests)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import copy

from data.cifar import get_cifar10_loaders
from models.resnet import resnet56
from experiments.baselines.magnitude import MagnitudePruner
from metrics.sparsity import global_sparsity, print_sparsity_table
from training.trainer import evaluate


SPARSITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9]   # 0% → 30% → 50% → 70% → 90%
CHECKPOINT_PATH = "./checkpoints/resnet56_best.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: str, device: torch.device) -> nn.Module:
    model = resnet56(num_classes=10).to(device)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {path}  (val_acc={ckpt.get('val_acc', '?'):.2f}%)")
    else:
        print(f"WARNING: checkpoint not found at {path}. Using random weights (for smoke test).")
    return model


def run_magnitude_sweep(use_wandb: bool = False):
    """
    Sweep over SPARSITY_LEVELS, prune-and-eval at each level.
    Prints a results table and optionally logs to W&B.
    """
    _, _, test_loader = get_cifar10_loaders(
        data_dir="./data/downloads",
        batch_size=256,
        num_workers=0,  # 0 = main process only (Windows shared memory fix)
    )
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project="fanp", name="magnitude_pruning_sweep")

    print(f"\n{'Sparsity':>10} | {'Test Acc':>10} | {'Test Loss':>10} | {'Non-zero Params':>16}")
    print("-" * 55)

    results = []
    for sparsity in SPARSITY_LEVELS:
        # Load fresh copy each time so pruning levels are independent
        model  = load_model(CHECKPOINT_PATH, DEVICE)
        pruner = MagnitudePruner(model)

        if sparsity > 0.0:
            info = pruner.prune(sparsity=sparsity, scope="global")
            actual_sp = info["overall_sparsity"]
        else:
            actual_sp = 0.0

        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, split="test")

        total_params  = sum(p.numel() for p in model.parameters() if "conv" in p.__class__.__name__ or True)
        nonzero      = sum((p != 0).sum().item() for p in model.parameters())

        print(f"  {actual_sp*100:>7.1f}%  | {test_acc:>9.2f}% | {test_loss:>10.4f} | {nonzero:>16,}")
        results.append({"sparsity": actual_sp, "test_acc": test_acc, "test_loss": test_loss})

        if wandb_run:
            wandb_run.log({"magnitude/sparsity": actual_sp, "magnitude/test_acc": test_acc})

        if sparsity > 0.0:
            print_sparsity_table(model)

    if wandb_run:
        wandb_run.finish()

    return results


if __name__ == "__main__":
    results = run_magnitude_sweep(use_wandb=False)

    print("\n=== SUMMARY: Magnitude Pruning Baseline ===")
    print(f"{'Sparsity':>10} | {'Test Acc':>10}")
    print("-" * 25)
    for r in results:
        print(f"  {r['sparsity']*100:>7.1f}%  | {r['test_acc']:>9.2f}%")
