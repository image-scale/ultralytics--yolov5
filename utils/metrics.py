# YOLOv5 metrics

import numpy as np
import torch


def fitness(x):
    """Calculate model fitness as weighted sum of metrics."""
    raise NotImplementedError


def smooth(y, f=0.05):
    """Smooth values using a moving filter."""
    raise NotImplementedError


def compute_ap(recall, precision):
    """Compute average precision from recall and precision curves."""
    raise NotImplementedError


def box_iou(box1, box2):
    """Calculate IoU between box arrays."""
    raise NotImplementedError


class ConfusionMatrix:
    """Confusion matrix for object detection."""
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        raise NotImplementedError
