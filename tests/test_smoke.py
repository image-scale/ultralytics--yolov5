"""Smoke tests for YOLOv5 modules.

These tests verify that core modules can be imported and basic functionality
works without GPU or pretrained model downloads.
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Verify that key project modules import cleanly."""

    def test_import_utils_init(self):
        from utils import TryExcept, emojis, threaded
        assert callable(emojis)
        assert callable(threaded)

    def test_import_utils_metrics(self):
        from utils.metrics import fitness, smooth, compute_ap, box_iou, ConfusionMatrix
        assert callable(fitness)
        assert callable(smooth)
        assert callable(compute_ap)
        assert callable(box_iou)

    def test_import_utils_general(self):
        from utils.general import (
            non_max_suppression, xyxy2xywh, xywh2xyxy,
            check_img_size, is_ascii, intersect_dicts
        )
        assert callable(non_max_suppression)
        assert callable(xyxy2xywh)

    def test_import_models_common(self):
        from models.common import Conv, Bottleneck, C3, SPPF
        assert Conv is not None
        assert C3 is not None

    def test_import_models_yolo(self):
        from models.yolo import Model, Detect
        assert Model is not None
        assert Detect is not None


class TestMetrics:
    """Test metric computation functions."""

    def test_smooth(self):
        import numpy as np
        from utils.metrics import smooth

        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = smooth(y, f=0.05)
        assert result.shape == y.shape
        assert result.dtype == float

    def test_compute_ap(self):
        import numpy as np
        from utils.metrics import compute_ap

        recall = np.linspace(0, 1, 10)
        precision = np.ones(10) * 0.8
        ap, mpre, mrec = compute_ap(recall, precision)
        assert 0.0 <= ap <= 1.0
        assert len(mpre) > 0
        assert len(mrec) > 0

    def test_box_iou(self):
        import torch
        from utils.metrics import box_iou

        # Two identical boxes should have IoU = 1.0
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        iou = box_iou(boxes, boxes)
        assert iou.shape == (1, 1)
        assert abs(float(iou[0, 0]) - 1.0) < 1e-5

    def test_box_iou_no_overlap(self):
        import torch
        from utils.metrics import box_iou

        box_a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box_b = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        iou = box_iou(box_a, box_b)
        assert abs(float(iou[0, 0])) < 1e-5

    def test_confusion_matrix_init(self):
        from utils.metrics import ConfusionMatrix
        cm = ConfusionMatrix(nc=5, conf=0.25, iou_thres=0.45)
        assert cm.matrix.shape == (6, 6)
        assert cm.nc == 5

    def test_fitness(self):
        import numpy as np
        from utils.metrics import fitness

        # Simulate metrics array: [P, R, mAP@0.5, mAP@0.5:0.95]
        x = np.array([[0.8, 0.7, 0.6, 0.5]])
        result = fitness(x)
        assert result.shape == (1,)
        # fitness = 0.0*P + 0.0*R + 0.1*mAP50 + 0.9*mAP50:95 = 0.1*0.6 + 0.9*0.5 = 0.51
        assert abs(float(result[0]) - 0.51) < 1e-5


class TestGeneralUtils:
    """Test pure utility functions from utils.general."""

    def test_is_ascii(self):
        from utils.general import is_ascii
        assert is_ascii("hello") is True
        assert is_ascii("") is True

    def test_check_img_size(self):
        from utils.general import check_img_size
        # Standard sizes should pass through
        result = check_img_size(640)
        assert result == 640

        # Non-divisible by stride should be adjusted
        result = check_img_size(641, s=32)
        assert result % 32 == 0

    def test_xyxy2xywh(self):
        import torch
        from utils.general import xyxy2xywh

        boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        xywh = xyxy2xywh(boxes)
        assert xywh.shape == (1, 4)
        # center should be (2, 2) and width/height (4, 4)
        assert abs(float(xywh[0, 0]) - 2.0) < 1e-5
        assert abs(float(xywh[0, 1]) - 2.0) < 1e-5
        assert abs(float(xywh[0, 2]) - 4.0) < 1e-5
        assert abs(float(xywh[0, 3]) - 4.0) < 1e-5

    def test_xywh2xyxy(self):
        import torch
        from utils.general import xywh2xyxy

        boxes = torch.tensor([[2.0, 2.0, 4.0, 4.0]])
        xyxy = xywh2xyxy(boxes)
        assert xyxy.shape == (1, 4)
        # x1=0, y1=0, x2=4, y2=4
        assert abs(float(xyxy[0, 0]) - 0.0) < 1e-5
        assert abs(float(xyxy[0, 2]) - 4.0) < 1e-5

    def test_intersect_dicts(self):
        import torch
        from utils.general import intersect_dicts

        # intersect_dicts compares tensor shapes, so values must be tensors
        t = torch.zeros(3)
        da = {"a": t, "b": t, "c": t}
        db = {"a": t, "b": t, "d": t}
        result = intersect_dicts(da, db)
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_intersect_dicts_with_exclude(self):
        import torch
        from utils.general import intersect_dicts

        t = torch.zeros(3)
        da = {"a": t, "b": t, "c": t}
        db = {"a": t, "b": t}
        result = intersect_dicts(da, db, exclude=("a",))
        assert "a" not in result
        assert "b" in result


class TestModelBuilding:
    """Test that YOLOv5 model architectures can be built on CPU."""

    def test_build_yolov5n(self):
        import torch
        from models.yolo import Model

        model = Model("models/yolov5n.yaml", ch=3, nc=80)
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_build_yolov5s(self):
        import torch
        from models.yolo import Model

        model = Model("models/yolov5s.yaml", ch=3, nc=80)
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000  # yolov5s has ~7M params

    def test_model_forward_pass(self):
        import torch
        from models.yolo import Model

        model = Model("models/yolov5n.yaml", ch=3, nc=80)
        model.eval()
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        # Detection model returns (predictions, feature_maps) during eval
        assert out is not None

    def test_model_nc_classes(self):
        from models.yolo import Model

        model = Model("models/yolov5n.yaml", ch=3, nc=10)
        # Verify number of detection classes
        detect_layer = model.model[-1]
        assert detect_layer.nc == 10
