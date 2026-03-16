"""
FANP Composite Forgetting Score.

Combines all three importance signals into a single per-weight score:

    S_i = α · F̂_i  +  β · σ²(g_i)  +  γ · ΔL_i

Where:
  F̂_i    = Empirical Fisher (mean squared gradient)      from fisher.py
  σ²(g_i) = Gradient Variance over window K              from gradient_variance.py
  ΔL_i   = Loss Spike on weight removal                  from loss_spike.py

Each component is independently min-max normalised to [0, 1] before
combining, so the α/β/γ weights have consistent meaning regardless of
the raw scale differences between signals.

Higher score = MORE important = should be kept.
Lower score  = LESS important = candidate for pruning.

Fixed weights (α=0.5, β=0.3, γ=0.2) are used by default, matching the
project document.  These can be overridden at construction time — Phase 4
will meta-learn them.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

from .fisher import EmpiricalFisher
from .gradient_variance import GradientVariance
from .loss_spike import LossSpike


def _minmax_normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise tensor to [0, 1]. Returns zeros if range is too small."""
    lo, hi = t.min(), t.max()
    if (hi - lo).item() < eps:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo + eps)


class ForgettingScore:
    """
    Orchestrates Fisher + GradVar + LossSpike into the FANP Forgetting Score.

    Args:
        model:      The trained neural network.
        alpha:      Weight for Fisher term (default 0.5).
        beta:       Weight for Gradient Variance term (default 0.3).
        gamma:      Weight for Loss Spike term (default 0.2).
        window_K:   Sliding window size for GradVar.
        n_batches:  Batches used for LossSpike evaluation.
        device:     Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.5,
        beta:  float = 0.3,
        gamma: float = 0.2,
        window_K:   int = 50,
        n_batches:  int = 10,
        device: torch.device | None = None,
    ):
        self.model  = model
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self._device = device or next(model.parameters()).device

        self.fisher   = EmpiricalFisher(model, device=self._device)
        self.grad_var = GradientVariance(model, window_K=window_K, device=self._device)
        self.loss_spike = LossSpike(model, device=self._device, n_batches=n_batches)

    # ------------------------------------------------------------------
    def accumulate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        """
        Process one batch through all three sub-estimators.
        Call this inside a data loader loop before calling compute().
        """
        self.fisher.accumulate(self.model, inputs, targets, criterion)
        self.grad_var.accumulate(self.model, inputs, targets, criterion)
        self.loss_spike.accumulate(self.model, inputs, targets, criterion)

    # ------------------------------------------------------------------
    def compute(self, criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute the composite Forgetting Score for every conv weight.

        Returns:
            dict[layer_name -> Tensor] — same shape as each conv weight.
            Higher = more important.
        """
        f_scores  = self.fisher.scores()
        gv_scores = self.grad_var.scores()
        ls_scores = self.loss_spike.scores(criterion=criterion)

        composite: Dict[str, torch.Tensor] = {}
        all_keys = set(f_scores) | set(gv_scores) | set(ls_scores)

        for name in all_keys:
            f  = _minmax_normalize(f_scores.get( name, torch.zeros(1)).to(self._device))
            gv = _minmax_normalize(gv_scores.get(name, torch.zeros(1)).to(self._device))
            ls = _minmax_normalize(ls_scores.get(name, torch.zeros(1)).to(self._device))

            composite[name] = self.alpha * f + self.beta * gv + self.gamma * ls

        return composite

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all sub-estimators (call between pruning rounds)."""
        self.fisher.reset()
        self.grad_var.reset()
        self.loss_spike.reset()

    # ------------------------------------------------------------------
    def summary(self, scores: Dict[str, torch.Tensor]) -> None:
        """Print a per-layer summary of mean forgetting scores."""
        print(f"\n{'Layer':<50} {'Mean FS':>10} {'Min FS':>10} {'Max FS':>10}")
        print("-" * 80)
        for name, s in sorted(scores.items()):
            print(f"  {name:<48} {s.mean().item():>10.4f} {s.min().item():>10.4f} {s.max().item():>10.4f}")
        all_vals = torch.cat([s.flatten() for s in scores.values()])
        print("-" * 80)
        print(f"  {'OVERALL':<48} {all_vals.mean().item():>10.4f} {all_vals.min().item():>10.4f} {all_vals.max().item():>10.4f}")
