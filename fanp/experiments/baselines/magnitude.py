"""
Magnitude Pruning Baseline (Han et al., 2015).

Strategy: rank all conv weights globally by |w|.
          Zero out the bottom `sparsity` fraction by applying a binary mask.

This is UNSTRUCTURED pruning — individual weights are zeroed, not filters.
It is the simplest and most common baseline for any pruning paper.

Usage:
    from experiments.baselines.magnitude import MagnitudePruner
    pruner = MagnitudePruner(model)
    pruner.prune(sparsity=0.5)      # remove 50% of weights
    acc = evaluate(model, ...)
    pruner.remove_masks()           # clean up hooks for export
"""
import torch
import torch.nn as nn


class MagnitudePruner:
    """
    Global unstructured magnitude pruning using PyTorch's built-in mask API.

    All Conv2d weight tensors across the network are ranked together
    (global ranking) and the lowest |w| fraction is zeroed out.

    Attributes
    ----------
    model      : the neural network to prune
    masks      : dict[name → binary mask tensor] set after calling prune()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.masks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Core pruning
    # ------------------------------------------------------------------

    def prune(self, sparsity: float, scope: str = "global") -> dict[str, float]:
        """
        Apply magnitude pruning at the given sparsity level.

        Parameters
        ----------
        sparsity : float
            Fraction of weights to zero out. e.g. 0.5 = remove 50%.
        scope : "global" or "local"
            global → single threshold across all layers (default, matches Han 2015)
            local  → each layer pruned to the same sparsity independently

        Returns
        -------
        info : dict with actual sparsity achieved per layer and overall.
        """
        assert 0.0 < sparsity < 1.0, "sparsity must be in (0, 1)"

        # Remove old masks before applying new ones
        self._remove_masks_from_params()
        self.masks.clear()

        if scope == "global":
            return self._prune_global(sparsity)
        else:
            return self._prune_local(sparsity)

    def _prune_global(self, sparsity: float) -> dict:
        """Single threshold computed across all conv weight tensors."""
        # 1. Collect all conv weights into one flat tensor
        all_weights = []
        param_list  = []   # (name, param) for each conv weight

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                all_weights.append(w.abs().flatten())
                param_list.append((name, module))

        if not param_list:
            raise ValueError("No Conv2d layers found in model.")

        # 2. Find the global threshold at the given percentile
        all_flat  = torch.cat(all_weights)                          # 1-D tensor of all |w|
        threshold = torch.quantile(all_flat, sparsity).item()       # scalar cutoff

        # 3. Build and apply masks
        info = {"threshold": threshold, "layers": {}}
        for name, module in param_list:
            mask = (module.weight.data.abs() > threshold).float()   # 1 = keep, 0 = prune
            self.masks[name] = mask
            module.weight.data.mul_(mask)                           # zero out pruned weights

            layer_sparsity = 1.0 - mask.mean().item()
            info["layers"][name] = round(layer_sparsity, 4)

        # Overall sparsity
        total_params  = all_flat.numel()
        pruned_params = sum((1 - m.float()).sum().item() for m in self.masks.values())
        info["overall_sparsity"] = round(pruned_params / total_params, 4)

        return info

    def _prune_local(self, sparsity: float) -> dict:
        """Each layer gets its own threshold — all layers reach the same sparsity."""
        info = {"layers": {}}
        total_w, total_pruned = 0, 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                w         = module.weight.data
                flat      = w.abs().flatten()
                threshold = torch.quantile(flat, sparsity).item()
                mask      = (w.abs() > threshold).float()

                self.masks[name] = mask
                module.weight.data.mul_(mask)

                layer_sparsity = 1.0 - mask.mean().item()
                info["layers"][name] = round(layer_sparsity, 4)

                total_w       += flat.numel()
                total_pruned  += int((mask == 0).sum().item())

        info["overall_sparsity"] = round(total_pruned / total_w, 4)
        return info

    # ------------------------------------------------------------------
    # Mask persistence — reapply masks after any weight update
    # ------------------------------------------------------------------

    def reapply_masks(self):
        """
        Re-zero pruned weights. Must be called after every optimizer.step()
        during fine-tuning so that gradients don't 'revive' pruned weights.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                module.weight.data.mul_(self.masks[name])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _remove_masks_from_params(self):
        """Reset: remove stored masks (does NOT restore pruned values)."""
        self.masks.clear()

    def remove_masks(self):
        """Public cleanup — call before ONNX export or model hand-off."""
        self._remove_masks_from_params()

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def current_sparsity(self) -> float:
        """Measure actual proportion of zero weights in conv layers."""
        total, zeros = 0, 0
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                w      = module.weight.data
                total += w.numel()
                zeros += (w == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    def sparsity_per_layer(self) -> dict[str, float]:
        """Return zero-weight fraction for each conv layer."""
        result = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                result[name] = (w == 0).sum().item() / w.numel()
        return result
