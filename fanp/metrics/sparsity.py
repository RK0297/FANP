"""
Sparsity tracking utilities.
Reports zero-weight fraction globally and per layer.
"""
import torch
import torch.nn as nn


def global_sparsity(model: nn.Module) -> float:
    """Overall fraction of zero weights across all Conv2d + Linear layers."""
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w      = module.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def layer_sparsity(model: nn.Module) -> dict[str, float]:
    """Per-layer zero-weight fraction for all Conv2d layers."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            result[name] = (w == 0).sum().item() / w.numel()
    return result


def print_sparsity_table(model: nn.Module):
    """Print a formatted table of per-layer and overall sparsity."""
    table = layer_sparsity(model)
    print(f"\n{'Layer':<45} {'Sparsity':>10}")
    print("-" * 57)
    for name, sp in table.items():
        bar = "█" * int(sp * 20)
        print(f"{name:<45} {sp*100:>8.2f}%  {bar}")
    print("-" * 57)
    print(f"{'OVERALL':<45} {global_sparsity(model)*100:>8.2f}%\n")


def count_nonzero_params(model: nn.Module) -> tuple[int, int]:
    """Returns (nonzero_params, total_params) for Conv2d + Linear layers."""
    total, nonzero = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w        = module.weight.data
            total   += w.numel()
            nonzero += (w != 0).sum().item()
    return nonzero, total
