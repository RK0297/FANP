"""
Structured (filter-level) FANP Pruner.

Instead of zeroing individual weights (unstructured), this pruner physically
removes entire convolutional filters from the network.  The result is a
smaller model with real FLOPs and parameter-count reductions — unlike
unstructured pruning, the speedup is measurable at inference time.

Challenge: ResNet skip connections create *dependency chains*.
Pruning filters in one layer forces compatible changes in adjacent layers
(e.g. the residual shortcut must match the main branch channel count).
torch-pruning's DependencyGraph resolves this automatically.

Filter importance
-----------------
We aggregate the FANP weight-level ForgettingScore to the filter level:

    I_f = mean_{c, kH, kW}  S[f, c, kH, kW]

where S is the per-weight ForgettingScore from composite.py.
Higher I_f = this output channel encodes more "forgettable" knowledge = KEEP.
Lowest I_f filters = safe to remove.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional

import torch_pruning as tp

from pruning.importance.composite import ForgettingScore


# ---------------------------------------------------------------------------
# Custom torch-pruning Importance class
# ---------------------------------------------------------------------------

class FANPFilterImportance(tp.importance.Importance):
    """
    torch-pruning Importance backed by pre-computed FANP filter scores.

    Assigned filter importance = mean FANP ForgettingScore across all
    weights in that output channel.  Coupled layers in the same dependency
    group are handled by averaging their per-channel scores.

    Parameters
    ----------
    layer_scores : dict[str, Tensor]
        Maps the module's dotted name (e.g. ``"layer1.0.conv1"``) to a
        1-D tensor of shape ``[out_channels]`` holding per-filter scores.
    """

    def __init__(self, layer_scores: Dict[str, torch.Tensor]) -> None:
        self._layer_scores = layer_scores

    @torch.no_grad()
    def __call__(
        self,
        group: tp.dependency.Group,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Return per-channel importance for the given pruning group.

        The group bundles all layers that must be pruned together (e.g. the
        two conv layers sharing a residual shortcut).  We average FANP scores
        across the group members that we have scores for.
        """
        group_imp: List[torch.Tensor] = []

        for dep, idxs in group:
            idxs  = list(idxs)
            layer = dep.target.module

            if not isinstance(layer, nn.Conv2d):
                continue

            # dep.handler is a bound method; compare by function name
            handler_name = getattr(
                getattr(dep.handler, "__func__", dep.handler),
                "__name__", ""
            )
            if handler_name not in ("prune_out_channels", "prune_in_channels"):
                continue

            # dep.target.name has format "layer1.0.conv1 (Conv2d(...))"
            # Strip the repr suffix to get just the dotted module path
            raw_name = dep.target.name
            name = raw_name.split(" (")[0].strip()

            if name not in self._layer_scores:
                continue

            channel_scores = self._layer_scores[name]   # [out_channels]

            # Guard against index out of range (edge case during group resolution)
            valid_idxs = [i for i in idxs if i < channel_scores.shape[0]]
            if not valid_idxs:
                continue

            group_imp.append(channel_scores[valid_idxs])

        if not group_imp:
            return None

        # Mean across all group members → shape [n_candidate_channels]
        try:
            stacked = torch.stack(group_imp, dim=0)
        except RuntimeError:
            # Lengths differ across group members (unusual) — fall back to first
            return group_imp[0]

        return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Structured FANP Pruner
# ---------------------------------------------------------------------------

class StructuredFANPPruner:
    """
    Structured filter pruner using FANP ForgettingScore for filter ranking.

    Computes per-filter importance by aggregating weight-level FANP scores,
    then uses torch-pruning's MetaPruner to safely remove the least important
    filters while respecting ResNet dependency constraints.

    Parameters
    ----------
    model : nn.Module
        Trained model to prune (modified in place by ``prune()``).
    criterion : nn.Module
        Loss function used for importance scoring.
    device : torch.device
        Computation device.
    alpha : float
        Fisher weight in the composite ForgettingScore.
    beta : float
        GradientVariance weight.
    gamma : float
        Taylor criterion weight.
    acc_batches : int
        Number of mini-batches used to accumulate importance scores.
    window_K : int
        Sliding window size for GradientVariance.
    """

    def __init__(
        self,
        model:       nn.Module,
        criterion:   nn.Module,
        device:      torch.device,
        alpha:       float = 0.5,
        beta:        float = 0.3,
        gamma:       float = 0.2,
        acc_batches: int   = 20,
        window_K:    int   = 50,
    ) -> None:
        self.model        = model
        self.criterion    = criterion
        self.device       = device
        self._acc_batches = acc_batches

        self._fs_engine = ForgettingScore(
            model,
            alpha=alpha, beta=beta, gamma=gamma,
            window_K=window_K,
            n_batches=acc_batches,
            device=device,
        )

    # ------------------------------------------------------------------
    def _compute_filter_scores(self, loader) -> Dict[str, torch.Tensor]:
        """
        Accumulate ForgettingScore and aggregate to filter level.

        Returns
        -------
        dict[str, Tensor]
            Maps module name (e.g. ``"layer1.0.conv1"``) to a 1-D tensor of
            shape ``[out_channels]`` with per-filter importance scores.
        """
        self._fs_engine.reset()
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= self._acc_batches:
                break
            self._fs_engine.accumulate(inputs, targets, self.criterion)

        # weight_scores: {param_name: [out_ch, in_ch, kH, kW]}
        weight_scores = self._fs_engine.compute(self.criterion)

        filter_scores: Dict[str, torch.Tensor] = {}
        for param_name, w_scores in weight_scores.items():
            if not param_name.endswith(".weight"):
                continue
            module_name = param_name[: -len(".weight")]
            if w_scores.dim() == 4:     # Conv2d: [out, in, kH, kW]
                filter_scores[module_name] = w_scores.mean(dim=[1, 2, 3])

        return filter_scores

    # ------------------------------------------------------------------
    def prune(
        self,
        loader,
        pruning_ratio:  float = 0.3,
        example_inputs: Optional[torch.Tensor] = None,
        ignored_layers: Optional[List[nn.Module]] = None,
    ) -> Dict:
        """
        Structurally prune the model's convolutional filters.

        Parameters
        ----------
        loader : DataLoader
            Data loader used for FANP importance scoring.
        pruning_ratio : float
            Fraction of output channels to remove per layer (e.g. 0.3 = 30 %).
        example_inputs : Tensor, optional
            A single batch used to trace the model for the dependency graph.
            Defaults to a CIFAR-10 compatible random tensor.
        ignored_layers : list[nn.Module], optional
            Layers to skip (typically the final classifier FC layer).

        Returns
        -------
        dict
            ``params_before``, ``params_after``, ``compression``,
            ``pruning_ratio``.
        """
        assert 0.0 < pruning_ratio < 1.0, "pruning_ratio must be in (0, 1)"

        if example_inputs is None:
            example_inputs = torch.randn(1, 3, 32, 32, device=self.device)

        if ignored_layers is None:
            ignored_layers = []
            for attr in ("fc", "linear", "classifier"):
                layer = getattr(self.model, attr, None)
                if layer is not None:
                    ignored_layers.append(layer)

        params_before = sum(p.numel() for p in self.model.parameters())

        print("Computing FANP filter importance scores for structured pruning...")
        filter_scores = self._compute_filter_scores(loader)
        print(f"  Scored {len(filter_scores)} conv layers.")

        importance = FANPFilterImportance(filter_scores)

        pruner = tp.pruner.MetaPruner(
            self.model,
            example_inputs,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()

        params_after = sum(p.numel() for p in self.model.parameters())
        compression  = params_before / max(params_after, 1)

        stats = {
            "params_before": params_before,
            "params_after":  params_after,
            "compression":   compression,
            "pruning_ratio": pruning_ratio,
        }
        print(
            f"Structured pruning complete | "
            f"params: {params_before:,} -> {params_after:,} "
            f"({compression:.2f}x compression, {pruning_ratio:.0%} channels removed)"
        )
        return stats
