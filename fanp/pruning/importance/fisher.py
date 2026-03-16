"""
Empirical Fisher Information estimator.

For each weight w_i in a conv layer, the empirical Fisher is:

    F̂_i = (1/N) * sum_n [ (∂L_n / ∂w_i)² ]

This is the expected squared gradient — it measures how sensitive the
loss is to perturbations of that weight. High Fisher = very important.

This is a diagonal approximation (one scalar per weight), which is
computationally cheap and works well in practice for pruning.

Reference: Molchanov et al., ICLR 2017; Theis et al., arXiv 2018.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict

from .base import ImportanceEstimator


class EmpiricalFisher(ImportanceEstimator):
    """
    Accumulates empirical Fisher (mean squared gradient) for every
    Conv2d weight in the model over multiple batches.

    Args:
        model:      The neural network (must have conv weight params).
        device:     Where to accumulate sums (default: same as model).
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        super().__init__(model)
        self._device = device or next(model.parameters()).device
        # Running sum of squared gradients and step count
        self._sum_sq_grad: Dict[str, torch.Tensor] = {}
        self._steps: int = 0
        self._init_buffers()

    # ------------------------------------------------------------------
    def _init_buffers(self):
        """Allocate zero buffers for each conv weight parameter."""
        self._sum_sq_grad.clear()
        self._steps = 0
        for name, param in self.model.named_parameters():
            if "conv" in name and "weight" in name:
                self._sum_sq_grad[name] = torch.zeros_like(
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
        Forward + backward one batch, accumulate squared gradients.
        Does NOT update model weights — purely observational.
        """
        model.eval()   # no dropout / BN noise during scoring
        inputs  = inputs.to(self._device)
        targets = targets.to(self._device)

        model.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._sum_sq_grad and param.grad is not None:
                    self._sum_sq_grad[name] += param.grad.detach().pow(2)

        self._steps += 1

    # ------------------------------------------------------------------
    def scores(self) -> Dict[str, torch.Tensor]:
        """
        Returns F̂_i = mean squared gradient for each weight.
        Shape of each tensor matches the corresponding weight tensor.
        """
        if self._steps == 0:
            raise RuntimeError("Call accumulate() at least once before scores().")
        return {
            name: sq / self._steps
            for name, sq in self._sum_sq_grad.items()
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear buffers — call before starting a fresh estimation pass."""
        self._init_buffers()
