# YOLOv5 YOLO-specific modules

import torch
import torch.nn as nn


class Detect(nn.Module):
    """YOLOv5 Detect head for detection models."""
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Model(nn.Module):
    """YOLOv5 model."""

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        raise NotImplementedError

    def forward(self, x, augment=False, profile=False, visualize=False):
        raise NotImplementedError

    def _forward_once(self, x, profile=False, visualize=False):
        raise NotImplementedError
