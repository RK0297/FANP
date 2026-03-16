"""
Entry point: train the ResNet-56 baseline on CIFAR-10.

Run from the fanp/ directory:
    python train_baseline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from training.trainer import train

# Inline config (mirrors configs/base.yaml — swap to Hydra in Phase 2)
cfg = {
    "seed": 42,
    "device": "cuda",

    "data": {
        "dataset":     "cifar10",
        "data_dir":    "./data/downloads",
        "batch_size":  128,
        "num_workers": 0,   # 0 = main process only (avoids Windows shared memory crash)
        "val_split":   0.1,
    },

    "model": {
        "arch":        "resnet56",
        "num_classes": 10,
    },

    "training": {
        "epochs":       200,
        "lr":           0.1,
        "momentum":     0.9,
        "weight_decay": 1e-4,
        "lr_milestones": [100, 150],
        "lr_gamma":     0.1,
    },

    "logging": {
        "use_wandb":    True,       # offline mode — logs saved locally in wandb/
        "project":      "fanp",
        "run_name":     "baseline_resnet56_cifar10",
        "log_interval": 50,
    },

    "checkpoint": {
        "save_dir":   "./checkpoints",
        "save_every": 10,
        "keep_best":  True,
    },

    # Resume is now auto-detected from resnet56_last.pth (latest epoch).
    # Set to None to train from scratch.
    "resume": None,
}

if __name__ == "__main__":
    train(cfg)
