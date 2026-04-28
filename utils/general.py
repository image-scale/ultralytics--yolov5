# YOLOv5 general utilities

import math
import numpy as np
import torch


def is_ascii(s=''):
    """Check if string contains only ASCII characters.

    Args:
        s: String to check

    Returns:
        True if string is ASCII, False otherwise
    """
    s = str(s)  # convert list, tuple, None, etc. to str
    return all(ord(c) < 128 for c in s)


def check_img_size(imgsz, s=32, floor=0):
    """Verify image size is a multiple of stride s.

    Args:
        imgsz: Image size (int or list)
        s: Stride
        floor: Minimum value

    Returns:
        New size that is a multiple of stride
    """
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        imgsz = list(imgsz)
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    return new_size


def make_divisible(x, divisor):
    """Make x evenly divisible by divisor.

    Args:
        x: Value to round
        divisor: Divisor

    Returns:
        Nearest value >= x that is divisible by divisor
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x):
    """Convert boxes from xyxy format to xywh format.

    xyxy: (x1, y1, x2, y2) = top-left and bottom-right corners
    xywh: (cx, cy, w, h) = center and dimensions

    Args:
        x: Tensor of shape [..., 4]

    Returns:
        Tensor with same shape in xywh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """Convert boxes from xywh format to xyxy format.

    xywh: (cx, cy, w, h) = center and dimensions
    xyxy: (x1, y1, x2, y2) = top-left and bottom-right corners

    Args:
        x: Tensor of shape [..., 4]

    Returns:
        Tensor with same shape in xyxy format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def intersect_dicts(da, db, exclude=()):
    """Intersect two dictionaries by matching keys and tensor shapes.

    Used for loading pretrained weights when shapes may differ.

    Args:
        da: First dictionary
        db: Second dictionary
        exclude: Keys to exclude

    Returns:
        Dictionary with intersecting keys and matching shapes
    """
    return {k: v for k, v in da.items()
            if k in db
            and not any(x in k for x in exclude)
            and v.shape == db[k].shape}


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, nm=0):
    """Apply non-maximum suppression to detection outputs.

    Args:
        prediction: Model predictions tensor
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: Class filter
        agnostic: Class-agnostic NMS
        multi_label: Multiple labels per box
        labels: Known labels for autolabeling
        max_det: Maximum detections per image
        nm: Number of masks

    Returns:
        List of detections per image after NMS
    """
    # For smoke tests, just return empty list (not tested)
    if prediction is None:
        return []

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence filter

        if not x.shape[0]:
            continue

        x[:, 5:5 + nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:5 + nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), x[i, 5 + nc:]), 1)
        else:
            conf, j = x[:, 5:5 + nc].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 5 + nc:]), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_det:
            x = x[x[:, 4].argsort(descending=True)[:max_det]]

        c = x[:, 5:6] * (0 if agnostic else 4096)  # class offsets
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output
