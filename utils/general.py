# YOLOv5 general utilities

import math
import numpy as np
import torch


def is_ascii(s=''):
    """Check if string is ASCII."""
    raise NotImplementedError


def check_img_size(imgsz, s=32, floor=0):
    """Verify image size is a multiple of stride."""
    raise NotImplementedError


def xyxy2xywh(x):
    """Convert boxes from xyxy to xywh format."""
    raise NotImplementedError


def xywh2xyxy(x):
    """Convert boxes from xywh to xyxy format."""
    raise NotImplementedError


def intersect_dicts(da, db, exclude=()):
    """Intersect two dictionaries by matching keys and shapes."""
    raise NotImplementedError


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, nm=0):
    """Apply non-max suppression to detection outputs."""
    raise NotImplementedError
