"""
Post-pruning recovery fine-tuner for FANP.

After each pruning round the model needs recovery fine-tuning to compensate
for accuracy lost when weights were zeroed.

Algorithm
---------
1. Run SGD for ``n_steps`` steps with a cosine-annealed learning rate.
2. After every optimizer.step(), re-apply the pruning masks so that
   zeroed weights stay at exactly zero throughout fine-tuning.
3. Evaluate on ``val_loader`` at regular intervals to track the recovery.
4. Return a RecoveryTrace with the step-by-step accuracy history.

Recovery slope interpretation
------------------------------
High slope  → pruned weights were not critical; model adapts quickly.
Flat slope  → pruning was too aggressive; remaining weights cannot compensate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecoveryTrace:
    """Records the accuracy trajectory during one fine-tuning session."""

    # List of (step, val_acc_percent) snapshots taken at eval_interval
    step_acc: List[Tuple[int, float]] = field(default_factory=list)

    # Accuracy before any fine-tuning begins
    initial_acc: float = 0.0

    # Accuracy after the final fine-tuning step
    final_acc: float = 0.0

    @property
    def accuracy_gain(self) -> float:
        """Total accuracy recovered (final - initial). May be negative."""
        return self.final_acc - self.initial_acc

    @property
    def recovery_slope(self) -> float:
        """Mean accuracy gain per fine-tuning step (percent / step)."""
        if len(self.step_acc) < 2:
            return 0.0
        total_steps = self.step_acc[-1][0] - self.step_acc[0][0]
        return self.accuracy_gain / max(total_steps, 1)


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------

class FineTuner:
    """
    Post-pruning recovery fine-tuner.

    Trains the pruned model for ``n_steps`` SGD steps with a cosine-annealed
    learning rate.  After every optimizer step the pruning masks are
    re-applied so that zeroed weights remain at exactly zero.

    Parameters
    ----------
    model : nn.Module
        The pruned model.  Fine-tuned **in place**.
    masks : dict[str, Tensor]
        Binary float masks keyed by parameter name.
        ``1`` = keep, ``0`` = pruned (stays zero).
    criterion : nn.Module
        Loss function (e.g. CrossEntropyLoss).
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model:     nn.Module,
        masks:     Dict[str, torch.Tensor],
        criterion: nn.Module,
        device:    torch.device,
    ) -> None:
        self.model     = model
        self.masks     = masks
        self.criterion = criterion
        self.device    = device

    # ------------------------------------------------------------------
    def _apply_masks(self) -> None:
        """Re-zero pruned weights after each optimizer step."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])

    @torch.no_grad()
    def _eval_acc(self, loader) -> float:
        """Top-1 accuracy on a DataLoader (returns percent, e.g. 92.5)."""
        self.model.eval()
        correct, total = 0, 0
        for inputs, targets in loader:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total   += targets.size(0)
        return 100.0 * correct / max(total, 1)

    # ------------------------------------------------------------------
    def fine_tune(
        self,
        train_loader,
        val_loader,
        n_steps:       int   = 200,
        lr:            float = 0.01,
        momentum:      float = 0.9,
        weight_decay:  float = 1e-4,
        eval_interval: int   = 50,
        verbose:       bool  = True,
    ) -> RecoveryTrace:
        """
        Run ``n_steps`` SGD update steps of recovery fine-tuning.

        Parameters
        ----------
        train_loader : DataLoader
            Training data stream.
        val_loader : DataLoader
            Validation data used for accuracy snapshots.
        n_steps : int
            Total number of optimizer steps to run.
        lr : float
            Initial learning rate.  Cosine-annealed to 0 over ``n_steps``.
        momentum : float
            SGD momentum coefficient.
        weight_decay : float
            L2 regularisation strength.
        eval_interval : int
            Steps between val_loader accuracy evaluations.
        verbose : bool
            Whether to print step-by-step progress.

        Returns
        -------
        RecoveryTrace
            Accuracy trajectory and summary statistics.
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_steps, 1), eta_min=0.0
        )

        trace = RecoveryTrace()
        trace.initial_acc = self._eval_acc(val_loader)
        trace.step_acc.append((0, trace.initial_acc))

        if verbose:
            print(f"  [FineTuner] start val_acc: {trace.initial_acc:.2f}%  "
                  f"({n_steps} steps, lr={lr})")

        step        = 0
        loader_iter = iter(train_loader)

        while step < n_steps:
            # Cycle through the loader if exhausted
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                inputs, targets = next(loader_iter)

            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss    = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep pruned weights exactly at zero
            self._apply_masks()

            step += 1

            if step % eval_interval == 0 or step == n_steps:
                acc = self._eval_acc(val_loader)
                trace.step_acc.append((step, acc))
                if verbose:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  [FineTuner] step {step:4d}/{n_steps}  "
                          f"val_acc: {acc:.2f}%  lr: {current_lr:.5f}")

        trace.final_acc = trace.step_acc[-1][1]
        return trace
