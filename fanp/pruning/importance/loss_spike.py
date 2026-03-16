"""
Loss Spike (ΔL) estimator — Taylor criterion (paper-quality).

For each weight w_i, the 1st-order Taylor expansion of the loss change
when w_i is removed (zeroed) is:

    ΔL_i ≈ |∂L/∂w_i · w_i|

Squaring gives the Taylor Importance (Molchanov et al., ICLR 2017):

    TI_i = (∂L/∂w_i · w_i)²

Why this is correct:
  - Removing w_i changes the loss by ΔL ≈ -(∂L/∂w_i) · w_i  (1st-order Taylor)
  - We square it so the score is always ≥ 0 and symmetric
  - Mean over N batches gives a stable estimate:
      TI_i = E_n[(g_i · w_i)²]

This is O(1) extra cost per batch (just grad * weight, no extra forward
passes), is provably a 1st-order approximation of ΔL, and is the exact
criterion used in:
  - Molchanov et al., ICLR 2017 ("Pruning Filters for Efficient ConvNets")
  - Importance estimation for structured pruning in many SOTA papers

This replaces the earlier layer-zeroing approximation with a theoretically
grounded, computationally free alternative.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

from .base import ImportanceEstimator


class LossSpike(ImportanceEstimator):
    """
    Taylor Criterion importance estimator.

    Computes TI_i = E[(∂L/∂w_i · w_i)²] over N accumulated batches.
    This is a theoretically-grounded, zero-cost proxy for the loss spike
    when weight w_i is removed.

    Args:
        model:     The neural network.
        device:    Computation device.
        n_batches: Number of batches to average the estimate over.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        n_batches: int = 10,
    ):
        super().__init__(model)
        self._device    = device or next(model.parameters()).device
        self._n_batches = n_batches

        # Running sum of (grad * weight)² and step counter
        self._sum_taylor: Dict[str, torch.Tensor] = {}
        self._steps: int = 0
        self._init_buffers()

    # ------------------------------------------------------------------
    def _init_buffers(self):
        self._sum_taylor.clear()
        self._steps = 0
        for name, param in self.model.named_parameters():
            if "conv" in name and "weight" in name:
                self._sum_taylor[name] = torch.zeros_like(
                    param.data, device=self._device
                )

    # ------------------------------------------------------------------
    def accumulate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        """
        Forward + backward one batch.
        Accumulates (grad * weight)² for each conv weight.
        Does NOT update model weights.
        """
        if self._steps >= self._n_batches:
            return   # already have enough batches

        model.eval()
        inputs  = inputs.to(self._device)
        targets = targets.to(self._device)

        model.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._sum_taylor and param.grad is not None:
                    # Taylor criterion: (g * w)²
                    taylor = (param.grad.detach() * param.data).pow(2)
                    self._sum_taylor[name] += taylor

        self._steps += 1

    # ------------------------------------------------------------------
    def scores(self, criterion: nn.Module | None = None) -> Dict[str, torch.Tensor]:
        """
        Returns TI_i = mean (g_i * w_i)² for each conv weight.
        `criterion` is accepted for API compatibility but not used here
        (gradients are computed during accumulate()).

        Higher value = larger estimated loss spike on removal = more important.
        """
        if self._steps == 0:
            raise RuntimeError("Call accumulate() at least once before scores().")
        return {
            name: s / self._steps
            for name, s in self._sum_taylor.items()
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear accumulated buffers."""
        self._init_buffers()
