"""
Main training loop for FANP baseline.
Trains ResNet-56 on CIFAR-10 to ~93% top-1 accuracy.
Supports W&B logging and checkpointing.
"""
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy as a percentage."""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return 100.0 * correct / target.size(0)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    wandb_run=None,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_acc   = accuracy(outputs, targets)
        total_loss += loss.item()
        total_acc  += batch_acc
        steps      += 1

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.1f}%")
            if wandb_run:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/acc":  batch_acc,
                    "epoch":      epoch,
                })

    return total_loss / steps, total_acc / steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss, total_acc, steps = 0.0, 0.0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        total_loss += loss.item()
        total_acc  += accuracy(outputs, targets)
        steps      += 1

    return total_loss / steps, total_acc / steps


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train(cfg: dict):
    """
    Full training pipeline driven by a config dict (mirrors base.yaml).
    Call directly or via train_baseline.py.
    """
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── W&B ─────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg["logging"]["use_wandb"]:
        try:
            import wandb
            import os
            os.environ["WANDB_MODE"] = "offline"   # logs locally, sync later with: wandb sync wandb/
            wandb_run = wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"]["run_name"],
                config=cfg,
            )
            print("W&B logging enabled (offline mode -- run 'wandb sync wandb/' to upload).")
        except Exception as e:
            print(f"W&B init failed ({e}) — continuing without logging.")
            wandb_run = None

    # ── Data ────────────────────────────────────────────────────────────
    from data.cifar import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["data"]["val_split"],
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ── Model ───────────────────────────────────────────────────────────
    from models.resnet import resnet56, resnet20
    arch    = cfg["model"]["arch"]
    model   = resnet56(num_classes=cfg["model"]["num_classes"]) if arch == "resnet56" else resnet20(num_classes=cfg["model"]["num_classes"])
    model   = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch} | Parameters: {n_params:,}")

    # Register gradient hooks (needed for Phase 2 Fisher/GradVar)
    model.register_gradient_hooks()

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg["training"]["lr_milestones"],
        gamma=cfg["training"]["lr_gamma"],
    )

    # ── Checkpoint dir ──────────────────────────────────────────────────
    ckpt_dir  = cfg["checkpoint"]["save_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_acc  = 0.0
    best_path = os.path.join(ckpt_dir, f"{arch}_best.pth")
    last_path = os.path.join(ckpt_dir, f"{arch}_last.pth")  # always latest epoch
    start_epoch = 1

    # ── Resume from checkpoint ──────────────────────────────────────────
    # Prefer the 'last' checkpoint (true latest epoch) over 'best' (highest val_acc)
    resume_path = cfg.get("resume", None)
    # Auto-detect: if last checkpoint exists, always prefer it
    if os.path.isfile(last_path):
        resume_path = last_path
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        # Restore best_acc from the best checkpoint (not the last, which may be lower)
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu")
            best_acc = best_ckpt.get("val_acc", 0.0)
        else:
            best_acc = ckpt.get("val_acc", 0.0)
        # Advance scheduler to the resumed epoch
        for _ in range(1, start_epoch):
            scheduler.step()
        print(f"Resumed from {resume_path} | epoch {ckpt['epoch']} | best_val_acc so far: {best_acc:.2f}%")

    # ── Training Loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, cfg["logging"]["log_interval"], wandb_run,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, split="val")
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.5f} | "
            f"Time: {elapsed:.1f}s"
        )

        if wandb_run:
            wandb_run.log({
                "epoch":      epoch,
                "val/loss":   val_loss,
                "val/acc":    val_acc,
                "train/epoch_loss": train_loss,
                "train/epoch_acc":  train_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":    val_acc,
                "cfg":        cfg,
            }, best_path)
            print(f"  >> New best val acc: {best_acc:.2f}% -- saved to {best_path}")

        # Always save latest checkpoint (overwrites each epoch — safe resume point)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc":         val_acc,
            "cfg":             cfg,
        }, last_path)

        # Periodic snapshot (kept forever, useful for analysis)
        if epoch % cfg["checkpoint"]["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{arch}_epoch{epoch}.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc}, ckpt_path)

    # ── Final test evaluation ────────────────────────────────────────────
    # Load best weights for final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, split="test")
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%  (best val: {best_acc:.2f}%)")

    if wandb_run:
        wandb_run.summary["test_acc"]  = test_acc
        wandb_run.summary["best_val_acc"] = best_acc
        wandb_run.finish()

    return model, test_acc
