"""
Entry point: train the ResNet-56 baseline on CIFAR-10.

Run from the fanp/ directory:
    python train_baseline.py
"""
import argparse
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from omegaconf import OmegaConf
from training.trainer import train


def _default_run_name() -> str:
    return f"baseline_resnet56_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def load_cfg(config_path: str) -> dict:
    """Load baseline training config from YAML via OmegaConf."""
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Expected dict config, got {type(cfg_dict)}")
    cfg_dict.setdefault("resume", None)
    logging_cfg = cfg_dict.setdefault("logging", {})
    if isinstance(logging_cfg, dict):
        run_name = logging_cfg.get("run_name", "baseline_resnet56_cifar10")
        if run_name == "baseline_resnet56_cifar10" or not run_name:
            logging_cfg["run_name"] = _default_run_name()
        logging_cfg["tags"] = ["baseline", "cifar10", "resnet56"]
    return cfg_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline from YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "base.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(load_cfg(args.config))
