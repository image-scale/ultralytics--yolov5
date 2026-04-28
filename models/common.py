# YOLOv5 common modules

import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Auto pad to 'same' shape outputs."""
    raise NotImplementedError


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def forward_fuse(self, x):
        raise NotImplementedError


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(self, c1, c2, k=5):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
