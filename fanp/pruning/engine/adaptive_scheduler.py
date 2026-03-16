"""
Adaptive Pruning Scheduler — FANP Engine.

Implements the adaptive pruning loop from the project document:

    while sparsity < target:
        1. Compute Forgetting Scores for all weights
        2. Compute mean FS across all layers
        3. If mean_FS > tau  →  halve the pruning rate (model is forgetting fast)
           Else              →  use base_rate
        4. Zero out the bottom-scoring fraction of weights (unstructured masking)
        5. (optional) recovery fine-tuning phase

The scheduler uses UNSTRUCTURED pruning (individual weight masking) by
default — Phase 3 will extend this to structured filter removal.

Key design:
  - Scores are computed FRESH each round over `acc_batches` batches
  - Pruning is cumulative: masks from previous rounds are preserved
  - Rate adapts per round, not per layer, for simplicity
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Callable

from pruning.importance.composite import ForgettingScore
from metrics.sparsity import global_sparsity


class AdaptivePruningScheduler:
    """
    FANP adaptive pruning scheduler.

    Args:
        model:          Trained model to prune (modified in place).
        criterion:      Loss function used for importance scoring.
        base_rate:      Fraction of remaining weights to prune per round (default 0.1).
        tau:            Mean FS threshold — if exceeded, rate is halved (default 0.5).
        alpha:          Fisher weight in composite score.
        beta:           GradVar weight in composite score.
        gamma:          LossSpike weight in composite score.
        window_K:       GradVar window size.
        acc_batches:    Batches per scoring pass.
        device:         Computation device.
    """

    def __init__(
        self,
        model:       nn.Module,
        criterion:   nn.Module,
        base_rate:   float = 0.10,
        tau:         float = 0.50,
        alpha:       float = 0.50,
        beta:        float = 0.30,
        gamma:       float = 0.20,
        window_K:    int   = 50,
        acc_batches: int   = 10,
        device: torch.device | None = None,
    ):
        self.model      = model
        self.criterion  = criterion
        self.base_rate  = base_rate
        self.tau        = tau
        self._device    = device or next(model.parameters()).device

        self.fs_engine = ForgettingScore(
            model,
            alpha=alpha, beta=beta, gamma=gamma,
            window_K=window_K, n_batches=acc_batches,
            device=self._device,
        )

        # Persistent binary masks: 1 = keep, 0 = pruned
        self._masks: Dict[str, torch.Tensor] = {}
        self._init_masks()

    # ------------------------------------------------------------------
    def _init_masks(self):
        """Initialise all-ones masks for each conv weight."""
        self._masks.clear()
        for name, param in self.model.named_parameters():
            if "conv" in name and "weight" in name:
                self._masks[name] = torch.ones_like(param.data, device=self._device)

    # ------------------------------------------------------------------
    def _apply_masks(self):
        """Zero out pruned weights according to current masks (no_grad)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._masks:
                    param.data.mul_(self._masks[name])

    # ------------------------------------------------------------------
    def _prune_round(self, scores: Dict[str, torch.Tensor], rate: float) -> int:
        """
        Zero out the lowest-scoring `rate` fraction of currently LIVE weights.
        Updates self._masks in place.

        Returns:
            Number of weights newly pruned this round.
        """
        # Gather all live scores (where mask == 1) into a flat tensor
        live_scores: List[torch.Tensor] = []
        for name, score in scores.items():
            if name in self._masks:
                live_mask  = self._masks[name].bool()
                live_scores.append(score[live_mask].flatten())

        if not live_scores:
            return 0

        all_live = torch.cat(live_scores)
        n_prune  = max(1, int(rate * all_live.numel()))
        threshold = all_live.kthvalue(n_prune).values.item()

        newly_pruned = 0
        for name, score in scores.items():
            if name not in self._masks:
                continue
            # Prune weights that are live AND score <= threshold
            prune_candidates = (score <= threshold) & self._masks[name].bool()
            self._masks[name][prune_candidates] = 0
            newly_pruned += prune_candidates.sum().item()

        return newly_pruned

    # ------------------------------------------------------------------
    def prune_to_target(
        self,
        loader,
        target_sparsity: float,
        max_rounds: int = 20,
        recovery_fn: Callable | None = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Main pruning loop.  Iteratively scores and prunes until target sparsity
        is reached or max_rounds is exceeded.

        Args:
            loader:           DataLoader used for scoring (a few batches only).
            target_sparsity:  Stop when global sparsity >= this value (0–1).
            max_rounds:       Safety cap on number of pruning rounds.
            recovery_fn:      Optional callable(model) called after each round
                              for fine-tuning / recovery.
            verbose:          Print round-by-round progress.

        Returns:
            List of per-round dicts with keys:
              round, sparsity, mean_fs, rate_used, n_pruned
        """
        history = []

        for round_idx in range(1, max_rounds + 1):
            current_sp = global_sparsity(self.model)
            if current_sp >= target_sparsity:
                if verbose:
                    print(f"Target sparsity {target_sparsity:.1%} reached. Stopping.")
                break

            # --- Score accumulation ---
            self.fs_engine.reset()
            for batch_idx, (inputs, targets) in enumerate(loader):
                if batch_idx >= self.fs_engine.loss_spike._n_batches:
                    break
                self.fs_engine.accumulate(inputs, targets, self.criterion)

            scores   = self.fs_engine.compute(self.criterion)
            mean_fs  = torch.cat([s.flatten() for s in scores.values()]).mean().item()

            # --- Adaptive rate ---
            rate = self.base_rate if mean_fs < self.tau else self.base_rate / 2

            # --- Prune ---
            n_pruned = self._prune_round(scores, rate)
            self._apply_masks()

            new_sp = global_sparsity(self.model)

            if verbose:
                print(
                    f"Round {round_idx:3d} | Sparsity: {new_sp:.3%} | "
                    f"mean_FS: {mean_fs:.4f} | rate: {rate:.3f} | "
                    f"pruned: {n_pruned:,}"
                )

            history.append({
                "round":    round_idx,
                "sparsity": new_sp,
                "mean_fs":  mean_fs,
                "rate_used": rate,
                "n_pruned": n_pruned,
            })

            # --- Optional recovery ---
            if recovery_fn is not None:
                recovery_fn(self.model)

        return history

    # ------------------------------------------------------------------
    @property
    def masks(self) -> Dict[str, torch.Tensor]:
        """Read-only access to current binary masks."""
        return self._masks

    def current_sparsity(self) -> float:
        """Global sparsity of the model under current masks."""
        return global_sparsity(self.model)
