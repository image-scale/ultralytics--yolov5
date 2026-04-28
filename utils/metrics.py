# YOLOv5 metrics

import numpy as np
import torch


def fitness(x):
    """Calculate model fitness as weighted sum of metrics.

    Args:
        x: Array with columns [P, R, mAP@0.5, mAP@0.5:0.95]

    Returns:
        Fitness score: 0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """Smooth values using box filter (moving average).

    Args:
        y: Array of values to smooth
        f: Filter fraction (smoothing window as fraction of array length)

    Returns:
        Smoothed array with same shape as y
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # padding elements
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # padded y
    return np.convolve(yp, np.ones(nf) / nf, mode='valid').astype(float)


def compute_ap(recall, precision):
    """Compute average precision from recall and precision curves.

    Uses 101-point interpolation (COCO-style).

    Args:
        recall: Recall curve
        precision: Precision curve

    Returns:
        (ap, interpolated_precision, interpolated_recall)
    """
    # Append sentinel values at the beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve using 101-point interpolation
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)  # integrate

    return ap, mpre, mrec


def box_iou(box1, box2):
    """Calculate IoU between two sets of boxes.

    Args:
        box1: Tensor of shape [N, 4] with xyxy format
        box2: Tensor of shape [M, 4] with xyxy format

    Returns:
        IoU matrix of shape [N, M]
    """
    # Get areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Inter-section
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


class ConfusionMatrix:
    """Confusion matrix for object detection."""

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initialize confusion matrix.

        Args:
            nc: Number of classes
            conf: Confidence threshold
            iou_thres: IoU threshold
        """
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))  # +1 for background

    def process_batch(self, detections, labels):
        """Process a batch of detections and labels."""
        pass  # Not tested

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        return tp[:-1], fp[:-1]  # exclude background class
