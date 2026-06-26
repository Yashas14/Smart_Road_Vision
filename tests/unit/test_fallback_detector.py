"""Unit tests for the classical OpenCV fallback detector."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from src.detection.detector import AnomalyDetector
from src.detection.fallback_detector import MODEL_VERSION, ClassicalAnomalyDetector
from src.detection.types import FrameResult


@pytest.fixture
def road_image() -> np.ndarray:
    """A synthetic grey road with a dark pothole, a blob and a crack."""
    rng = np.random.default_rng(7)
    img = np.full((480, 640, 3), 120, np.uint8)
    noise = rng.integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(img, (220, 300), 45, (35, 35, 35), -1)
    cv2.circle(img, (430, 180), 28, (60, 60, 60), -1)
    cv2.line(img, (100, 110), (300, 140), (40, 40, 40), 4)
    return img


def test_detects_anomalies_on_road_image(road_image: np.ndarray) -> None:
    detector = ClassicalAnomalyDetector()
    result = detector.detect_image(road_image)
    assert isinstance(result, FrameResult)
    assert result.count >= 1
    assert result.model_version == MODEL_VERSION


def test_detections_are_within_image_bounds(road_image: np.ndarray) -> None:
    detector = ClassicalAnomalyDetector()
    h, w = road_image.shape[:2]
    for det in detector.detect_image(road_image).detections:
        x1, y1, x2, y2 = det.bbox.xyxy()
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
        assert det.class_name in {"pothole", "hump", "crack", "road_degradation"}
        assert 0.0 <= det.confidence <= 1.0


def test_polygon_masks_are_populated(road_image: np.ndarray) -> None:
    detector = ClassicalAnomalyDetector()
    dets = detector.detect_image(road_image).detections
    assert any(d.polygon_mask for d in dets)


def test_uniform_image_yields_no_detections() -> None:
    detector = ClassicalAnomalyDetector()
    flat = np.full((300, 300, 3), 128, np.uint8)
    assert detector.detect_image(flat).count == 0


def test_max_detections_cap_is_respected(road_image: np.ndarray) -> None:
    detector = ClassicalAnomalyDetector(max_detections=1)
    assert detector.detect_image(road_image).count <= 1


def test_anomaly_detector_falls_back_without_ultralytics(
    road_image: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force import-time fallback path regardless of CI environment packages.
    monkeypatch.setitem(sys.modules, "ultralytics", object())
    detector = AnomalyDetector()
    detector.load()
    assert detector.using_fallback is True
    assert detector.is_loaded is True
    result = detector.detect_image(road_image)
    assert result.count >= 1
    assert result.model_version == MODEL_VERSION
