"""
Gradient Variance tracker.

For each weight w_i, gradient variance over a sliding window of K batches is:

    σ²(g_i) = Var[ ∂L/∂w_i ]  over last K steps
            = E[g²] - (E[g])²

Interpretation:
  - HIGH variance → this weight's gradient oscillates a lot across batches
    → the optimizer is actively fighting over its value
    → it is important for the task and should NOT be pruned
  - LOW variance → gradient is consistently near zero
    → weight is dormant → safe to prune

This captures something Fisher misses: a weight can have high mean squared
gradient (high Fisher) but low variance — it's important but stable.
GradVar surfaces weights that are needed and unstable to remove.

Reference: Hoffer et al., NeurIPS 2017; Frankle et al., ICLR 2020.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import Dict

from .base import ImportanceEstimator


class GradientVariance(ImportanceEstimator):
    """
    Tracks gradient variance over a sliding window of K recent batches
    for every Conv2d weight in the model.

    Args:
        model:      The neural network.
        window_K:   Number of recent batches to include in variance estimate.
        device:     Accumulation device.
    """

    def __init__(
        self,
        model: nn.Module,
        window_K: int = 50,
        device: torch.device | None = None,
    ):
        super().__init__(model)
        self._K      = window_K
        self._device = device or next(model.parameters()).device

        # deque of gradient tensors per layer — keeps last K only
        self._grad_window: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._K)
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
        Forward + backward one batch; push per-weight gradient to window.
        Does NOT update model weights.
        """
        model.eval()
        inputs  = inputs.to(self._device)
        targets = targets.to(self._device)

        model.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "conv" in name and "weight" in name and param.grad is not None:
                    # Store mean gradient per weight (detached, CPU to save VRAM)
                    self._grad_window[name].append(
                        param.grad.detach().cpu().clone()
                    )

    # ------------------------------------------------------------------
    def scores(self) -> Dict[str, torch.Tensor]:
        """
        Returns element-wise variance of gradients for each conv weight.
        Shape matches the corresponding weight tensor.
        Requires at least 2 accumulated batches.
        """
        result = {}
        for name, window in self._grad_window.items():
            if len(window) < 2:
                # Not enough data yet — return zeros (no variance signal)
                ref = next(
                    p for n, p in self.model.named_parameters() if n == name
                )
                result[name] = torch.zeros_like(ref.data)
                continue
            # Stack window: (K, *weight_shape)
            stacked = torch.stack(list(window), dim=0)   # (K, ...)
            result[name] = stacked.var(dim=0)             # element-wise variance
        return result

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear gradient windows."""
        self._grad_window.clear()
