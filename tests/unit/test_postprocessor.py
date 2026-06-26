"""Unit tests for :mod:`src.detection.postprocessor`."""

from __future__ import annotations

import numpy as np

from src.detection.postprocessor import (
    _iou,
    draw_annotations,
    non_max_suppression,
    normalize_coordinates,
)
from src.detection.types import SeverityLevel
from tests.conftest import make_detection


def test_iou_identical_boxes_is_one() -> None:
    a = make_detection(box=(0, 0, 100, 100))
    b = make_detection(box=(0, 0, 100, 100))
    assert _iou(a, b) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    a = make_detection(box=(0, 0, 10, 10))
    b = make_detection(box=(100, 100, 110, 110))
    assert _iou(a, b) == 0.0


def test_nms_on_empty_list() -> None:
    assert non_max_suppression([]) == []


def test_nms_suppresses_overlapping_same_class() -> None:
    high = make_detection(confidence=0.95, box=(0, 0, 100, 100))
    low = make_detection(confidence=0.50, box=(5, 5, 105, 105))
    kept = non_max_suppression([low, high], iou_threshold=0.4)
    assert len(kept) == 1
    assert kept[0].confidence == 0.95


def test_nms_keeps_different_classes_by_default() -> None:
    a = make_detection(class_name="pothole", class_id=0, box=(0, 0, 100, 100))
    b = make_detection(class_name="crack", class_id=1, box=(0, 0, 100, 100))
    kept = non_max_suppression([a, b], iou_threshold=0.4)
    assert len(kept) == 2


def test_nms_class_agnostic_suppresses_across_classes() -> None:
    a = make_detection(class_name="pothole", class_id=0, confidence=0.9, box=(0, 0, 100, 100))
    b = make_detection(class_name="crack", class_id=1, confidence=0.4, box=(0, 0, 100, 100))
    kept = non_max_suppression([a, b], iou_threshold=0.4, class_agnostic=True)
    assert len(kept) == 1
    assert kept[0].class_name == "pothole"


def test_normalize_coordinates_returns_unit_interval() -> None:
    dets = [make_detection(box=(0, 0, 320, 240))]
    norm = normalize_coordinates(dets, 640, 480)
    assert norm[0] == {"cx": 0.25, "cy": 0.25, "w": 0.5, "h": 0.5}


def test_draw_annotations_returns_new_image(synthetic_image: np.ndarray) -> None:
    det = make_detection(box=(10, 10, 60, 60))
    det.severity_level = SeverityLevel.CRITICAL
    out = draw_annotations(synthetic_image, [det])
    assert out.shape == synthetic_image.shape
    assert out is not synthetic_image


def test_draw_annotations_with_polygon_mask(synthetic_image: np.ndarray) -> None:
    det = make_detection(box=(10, 10, 80, 80))
    det.polygon_mask = [(10, 10), (80, 10), (80, 80), (10, 80)]
    out = draw_annotations(synthetic_image, [det], draw_masks=True)
    assert out.shape == synthetic_image.shape
