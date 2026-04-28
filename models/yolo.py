# YOLOv5 YOLO-specific modules

import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from models.common import Conv, C3, SPPF, Concat, Bottleneck


class Detect(nn.Module):
    """YOLOv5 Detect head for detection models."""
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initialize Detect head.

        Args:
            nc: Number of classes
            anchors: List of anchors for each detection layer
            ch: List of input channels for each detection layer
            inplace: Use in-place operations
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (classes + box + objectness)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace

    def forward(self, x):
        """Forward pass through detect layers."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Create a grid for anchor-based detection."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    """YOLOv5 model."""

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        """Initialize YOLOv5 model.

        Args:
            cfg: Model config file path or dict
            ch: Input channels
            nc: Number of classes (overrides config)
            anchors: Anchors (overrides config)
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)

        # Define model
        ch = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            self.yaml['anchors'] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([8., 16., 32.])  # Default strides for P3, P4, P5
            # Run a forward pass to verify shapes
            self.eval()
            with torch.no_grad():
                forward_result = self.forward(torch.zeros(1, ch, s, s))
            self.train()
            # Forward returns (predictions, feature_maps) in eval mode
            if isinstance(forward_result, tuple) and len(forward_result) > 1:
                feature_maps = forward_result[1]
                if isinstance(feature_maps, list) and len(feature_maps) > 0:
                    m.stride = torch.tensor([s / x.shape[-2] for x in feature_maps])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Forward pass.

        Args:
            x: Input tensor
            augment: Apply test-time augmentation
            profile: Profile model speed
            visualize: Visualize features

        Returns:
            Model predictions
        """
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        """Run forward pass once through the model."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _initialize_biases(self, cf=None):
        """Initialize Detect() biases."""
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(d, ch):
    """Parse YOLO model architecture from config dict.

    Args:
        d: Model config dict
        ch: List of input channels

    Returns:
        (model, save_list): nn.Sequential model and list of save indices
    """
    nc, gd, gw, anchors = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('anchors')
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings

        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in (Conv, Bottleneck, SPPF, C3):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in (C3,):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is nn.Upsample:
            c2 = ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    """Make x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):
    """Check anchor order matches stride order in Detect module."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)


import contextlib
