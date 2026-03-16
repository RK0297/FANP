"""
ResNet-56 for CIFAR-10 — original He et al. (2016) architecture.

Architecture:
  - Input: 3 x 32 x 32
  - Conv1: 3x3, 16 filters
  - Stage 1: 9 BasicBlocks, 16 filters  (32x32)
  - Stage 2: 9 BasicBlocks, 32 filters  (16x16, stride-2 on first block)
  - Stage 3: 9 BasicBlocks, 64 filters  (8x8,   stride-2 on first block)
  - GlobalAvgPool → FC(64, num_classes)
  Total layers: 2 + 3*(9*2) = 56

Gradient hooks are registered on every conv layer weight so that
we can track per-filter gradients for Fisher and GradVariance in Phase 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ---------------------------------------------------------------------------
# Building Block
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-layer residual block used in CIFAR ResNets."""

    expansion = 1  # output channels = planes * expansion

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Shortcut: only needed when dimensions change (stride > 1 or channels change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# ResNet for CIFAR (n blocks per stage → total layers = 6n + 2)
# ---------------------------------------------------------------------------

class ResNetCIFAR(nn.Module):
    """
    Generic CIFAR ResNet.
      n=9  → ResNet-56
      n=3  → ResNet-20  (lightweight for quick tests)
    """

    def __init__(self, n: int = 9, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # Three stages
        self.layer1 = self._make_stage(16, n, stride=1)
        self.layer2 = self._make_stage(32, n, stride=2)
        self.layer3 = self._make_stage(64, n, stride=2)

        self.fc = nn.Linear(64, num_classes)

        # Storage for gradient hooks (filled by register_gradient_hooks)
        self._grad_buffer: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._hook_handles: list = []

        self._init_weights()

    # -----------------------------------------------------------------------
    # Architecture helpers
    # -----------------------------------------------------------------------

    def _make_stage(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        strides  = [stride] + [1] * (n_blocks - 1)
        layers   = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming normal init for conv layers, batch norm init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)   # (B, 64, 1, 1)
        out = out.view(out.size(0), -1)        # (B, 64)
        out = self.fc(out)
        return out

    # -----------------------------------------------------------------------
    # Gradient Hooks — used by Phase 2 importance estimators
    # -----------------------------------------------------------------------

    def register_gradient_hooks(self):
        """
        Attach backward hooks to every Conv2d weight in the network.
        After each backward pass, the gradient for that parameter is appended
        to self._grad_buffer[layer_name] as a detached CPU tensor.

        Call clear_grad_buffer() periodically to avoid memory buildup.
        """
        self.remove_gradient_hooks()   # avoid double-registration
        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                # Capture `name` in closure
                def make_hook(n):
                    def hook(grad: torch.Tensor):
                        self._grad_buffer[n].append(grad.detach().cpu())
                    return hook
                handle = param.register_hook(make_hook(name))
                self._hook_handles.append(handle)

    def remove_gradient_hooks(self):
        """Detach all registered gradient hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def clear_grad_buffer(self):
        """Clear accumulated gradients from the buffer."""
        self._grad_buffer.clear()

    # -----------------------------------------------------------------------
    # Convenience: list prunable conv layers
    # -----------------------------------------------------------------------

    def prunable_layers(self) -> list[tuple[str, nn.Conv2d]]:
        """Returns [(name, module)] for every Conv2d that can be pruned."""
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Conv2d)]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet56(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-56 for CIFAR-10/100 (~0.85M parameters)."""
    return ResNetCIFAR(n=9, num_classes=num_classes)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 for quick smoke tests (~0.27M parameters)."""
    return ResNetCIFAR(n=3, num_classes=num_classes)
