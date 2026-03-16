"""
Abstract base class for all importance estimators in FANP.

Every estimator must implement:
  - accumulate(model, inputs, targets, criterion) — called each batch
  - scores() — returns dict[layer_name -> Tensor of per-weight importance]
  - reset() — clear accumulated state
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict


class ImportanceEstimator(ABC):
    """
    Base class for FANP importance estimators.

    Usage pattern:
        estimator = ConcreteEstimator(model)
        for inputs, targets in loader:
            estimator.accumulate(model, inputs, targets, criterion)
        scores = estimator.scores()   # dict[name -> Tensor]
        estimator.reset()
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def accumulate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> None:
        """Process one batch and accumulate internal statistics."""
        ...

    @abstractmethod
    def scores(self) -> Dict[str, torch.Tensor]:
        """
        Return per-weight importance scores for every prunable layer.

        Returns:
            dict mapping layer name (str) to a float Tensor of the same
            shape as that layer's weight parameter.
            Higher score = more important = should NOT be pruned.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state so the estimator can be reused."""
        ...
