"""Unit tests for detection types and post-processing (no heavy models)."""

from __future__ import annotations

import numpy as np
from tests.conftest import make_detection

from src.detection.postprocessor import (
    draw_annotations,
    non_max_suppression,
    normalize_coordinates,
)
from src.detection.types import BoundingBox, SeverityLevel


def test_bounding_box_geometry() -> None:
    box = BoundingBox(10, 20, 110, 220)
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20000
    assert box.centroid == (60.0, 120.0)


def test_bbox_xywhn_normalisation() -> None:
    box = BoundingBox(0, 0, 320, 240)
    cx, cy, w, h = box.xywhn(640, 480)
    assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


def test_nms_suppresses_overlapping_boxes() -> None:
    a = make_detection(confidence=0.9, box=(0, 0, 100, 100))
    b = make_detection(confidence=0.6, box=(5, 5, 105, 105))
    kept = non_max_suppression([a, b], iou_threshold=0.4, class_agnostic=True)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_nms_keeps_distinct_boxes() -> None:
    a = make_detection(confidence=0.9, box=(0, 0, 50, 50))
    b = make_detection(confidence=0.8, box=(300, 300, 350, 350))
    kept = non_max_suppression([a, b], iou_threshold=0.4, class_agnostic=True)
    assert len(kept) == 2


def test_normalize_coordinates_returns_dicts() -> None:
    det = make_detection(box=(0, 0, 320, 240))
    out = normalize_coordinates([det], 640, 480)
    assert out[0]["w"] == 0.5
    assert out[0]["h"] == 0.5


def test_draw_annotations_returns_new_image() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    det = make_detection(box=(10, 10, 90, 90))
    det.severity_level = SeverityLevel.HIGH
    annotated = draw_annotations(image, [det])
    assert annotated.shape == image.shape
    assert annotated is not image
    # Something was drawn (non-zero pixels exist).
    assert annotated.sum() > 0


def test_detection_to_dict_serialisable() -> None:
    det = make_detection()
    data = det.to_dict()
    assert data["class_name"] == "pothole"
    assert "bbox_xyxy" in data
    assert isinstance(data["timestamp"], str)
