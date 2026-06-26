"""Unit tests for shared detection dataclasses (:mod:`src.detection.types`)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.detection.types import (
    AnomalyDetection,
    BoundingBox,
    FrameResult,
    SeverityLevel,
    UrgencyTag,
)
from tests.conftest import make_detection


def test_bounding_box_dimensions() -> None:
    bbox = BoundingBox(10, 20, 110, 220)
    assert bbox.width == 100
    assert bbox.height == 200
    assert bbox.area == 20000
    assert bbox.centroid == (60, 120)
    assert bbox.xyxy() == (10, 20, 110, 220)


def test_bounding_box_clamps_negative_dimensions() -> None:
    # Inverted coordinates should not produce negative width/height.
    bbox = BoundingBox(100, 100, 10, 10)
    assert bbox.width == 0.0
    assert bbox.height == 0.0
    assert bbox.area == 0.0


def test_bounding_box_xywhn_normalisation() -> None:
    bbox = BoundingBox(0, 0, 100, 50)
    cx, cy, w, h = bbox.xywhn(200, 100)
    assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


def test_anomaly_detection_default_timestamp_is_utc() -> None:
    det = AnomalyDetection(0, "pothole", 0.9, BoundingBox(0, 0, 10, 10))
    assert det.timestamp.tzinfo is not None
    assert det.severity_level is SeverityLevel.LOW
    assert det.urgency is UrgencyTag.MONITOR


def test_anomaly_detection_to_dict_is_json_friendly() -> None:
    det = make_detection()
    det.timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    data = det.to_dict()
    assert data["bbox_xyxy"] == [100, 100, 200, 200]
    assert data["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert isinstance(data["severity_level"], str)
    assert isinstance(data["urgency"], str)


def test_frame_result_count_and_serialisation() -> None:
    frame = FrameResult(
        detections=[make_detection(), make_detection(class_name="crack")],
        image_width=640,
        image_height=480,
        processing_time_ms=10.0,
        model_version="v-test",
        road_condition_score=75.0,
    )
    assert frame.count == 2
    data = frame.to_dict()
    assert data["count"] == 2
    assert len(data["detections"]) == 2
    assert data["road_condition_score"] == 75.0
    assert data["model_version"] == "v-test"


def test_empty_frame_result_count_is_zero() -> None:
    frame = FrameResult([], 100, 100, 1.0, "v")
    assert frame.count == 0
    assert frame.to_dict()["detections"] == []
