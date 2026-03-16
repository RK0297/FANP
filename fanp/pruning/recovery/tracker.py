"""
Recovery Slope Tracker for FANP.

After each pruning round, calls FineTuner and measures the *recovery slope*:

    recovery_slope = delta_accuracy / delta_steps   [percent per step]

If the slope falls below ``slope_threshold``, the model is flagged as
OVERPRUNED — the remaining weights cannot compensate for what was removed.

Typical usage
-------------
    tracker = RecoveryTracker(model, scheduler.masks, criterion, device)

    # Called after each pruning round inside prune_to_target:
    metrics = tracker.measure(train_loader, val_loader,
                              round_idx=round_idx, sparsity=current_sp)
    if metrics.overpruned:
        break   # stop pruning further

    tracker.summary()   # print full history table
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional

from .fine_tuner import FineTuner, RecoveryTrace


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecoveryMetrics:
    """Summary of one post-pruning recovery session."""

    round_idx:      int
    sparsity:       float

    # Accuracy immediately after pruning (before any fine-tuning)
    acc_before:     float = 0.0

    # Accuracy after the fine-tuning session finishes
    acc_after:      float = 0.0

    # delta_acc / delta_steps averaged over the fine-tuning run
    recovery_slope: float = 0.0

    # True if recovery_slope < slope_threshold (model struggled to recover)
    overpruned:     bool  = False

    # Full step-by-step trace (None if not retained)
    trace: Optional[RecoveryTrace] = None


# ---------------------------------------------------------------------------
# RecoveryTracker
# ---------------------------------------------------------------------------

class RecoveryTracker:
    """
    Wraps FineTuner to measure and log recovery metrics after each pruning
    round.

    Parameters
    ----------
    model : nn.Module
        The pruned model that will be fine-tuned in place.
    masks : dict[str, Tensor]
        Binary pruning masks shared with the pruning scheduler.
        Since this is passed by reference, mask updates from the scheduler
        are automatically visible here.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Computation device.
    slope_threshold : float
        Minimum acceptable recovery slope (% accuracy / step).
        If ``recovery_slope < slope_threshold``, the round is flagged OVERPRUNED.
    n_steps : int
        Fine-tuning steps per recovery session.
    lr : float
        Fine-tuning learning rate (cosine-annealed to 0 over ``n_steps``).
    eval_interval : int
        Steps between validation accuracy evaluations during fine-tuning.
    """

    def __init__(
        self,
        model:           nn.Module,
        masks:           Dict[str, torch.Tensor],
        criterion:       nn.Module,
        device:          torch.device,
        slope_threshold: float = 0.005,     # 0.005 % per step
        n_steps:         int   = 200,
        lr:              float = 0.01,
        eval_interval:   int   = 50,
    ) -> None:
        self.model           = model
        self.masks           = masks
        self.criterion       = criterion
        self.device          = device
        self.slope_threshold = slope_threshold
        self.n_steps         = n_steps
        self.lr              = lr
        self.eval_interval   = eval_interval

        self.history: List[RecoveryMetrics] = []

    # ------------------------------------------------------------------
    def measure(
        self,
        train_loader,
        val_loader,
        round_idx: int,
        sparsity:  float,
        verbose:   bool = True,
    ) -> RecoveryMetrics:
        """
        Run one fine-tuning session and compute recovery metrics.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader
            Validation data used for accuracy measurement.
        round_idx : int
            Pruning round number — logged for analysis.
        sparsity : float
            Current model sparsity fraction — logged for analysis.
        verbose : bool
            Whether FineTuner prints step-by-step progress.

        Returns
        -------
        RecoveryMetrics
            Contains slope, overpruned flag, and the full RecoveryTrace.
        """
        ft = FineTuner(
            model=self.model,
            masks=self.masks,
            criterion=self.criterion,
            device=self.device,
        )
        trace = ft.fine_tune(
            train_loader=train_loader,
            val_loader=val_loader,
            n_steps=self.n_steps,
            lr=self.lr,
            eval_interval=self.eval_interval,
            verbose=verbose,
        )

        slope   = trace.recovery_slope
        metrics = RecoveryMetrics(
            round_idx      = round_idx,
            sparsity       = sparsity,
            acc_before     = trace.initial_acc,
            acc_after      = trace.final_acc,
            recovery_slope = slope,
            overpruned     = (slope < self.slope_threshold),
            trace          = trace,
        )
        self.history.append(metrics)
        self._print_round(metrics)
        return metrics

    # ------------------------------------------------------------------
    def _print_round(self, m: RecoveryMetrics) -> None:
        flag = "  [OVERPRUNED]" if m.overpruned else ""
        print(
            f"Recovery r{m.round_idx} | sparsity {m.sparsity*100:.1f}% | "
            f"acc {m.acc_before:.2f}% -> {m.acc_after:.2f}% "
            f"({m.acc_after - m.acc_before:+.2f}%) | "
            f"slope {m.recovery_slope:.5f}%/step{flag}"
        )

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a formatted table of all recovery rounds recorded so far."""
        if not self.history:
            print("RecoveryTracker: no rounds recorded yet.")
            return
        header = (f"{'Rnd':>4}  {'Sparsity':>9}  {'Before':>8}  "
                  f"{'After':>8}  {'Gain':>7}  {'Slope':>10}  {'Status':>12}")
        print(f"\n{header}")
        print("-" * 68)
        for m in self.history:
            status = "OVERPRUNED" if m.overpruned else "OK"
            gain   = m.acc_after - m.acc_before
            print(
                f"{m.round_idx:>4}  "
                f"{m.sparsity*100:>8.1f}%  "
                f"{m.acc_before:>7.2f}%  "
                f"{m.acc_after:>7.2f}%  "
                f"{gain:>+6.2f}%  "
                f"{m.recovery_slope:>9.5f}  "
                f"{status:>12}"
            )
        print()
