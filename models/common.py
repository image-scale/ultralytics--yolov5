# YOLOv5 common modules

import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Auto pad to 'same' shape outputs.

    Args:
        k: Kernel size
        p: Padding
        d: Dilation

    Returns:
        Padding value for 'same' convolution
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            d: Dilation
            act: Activation (True uses default SiLU, False/None means no activation)
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Forward pass: conv -> bn -> activation."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Forward pass without batch norm (for fused model)."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initialize Bottleneck.

        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Add residual connection
            g: Groups for 3x3 conv
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with optional residual."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3 module.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Use shortcut in Bottleneck
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass: concatenate two branches and merge."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5."""

    def __init__(self, c1, c2, k=5):
        """Initialize SPPF layer.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size for max pooling
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass: apply 3 sequential max pools and concatenate."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Initialize Concat layer.

        Args:
            dimension: Concatenation dimension
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass: concatenate input tensors."""
        return torch.cat(x, self.d)
