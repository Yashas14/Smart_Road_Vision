"""Shared pytest fixtures and test factories.

The detection components are heavy (YOLO/SAM2/MiDaS) and may be unavailable in
CI, so these fixtures provide a fake detector and a synthetic image so the test
suite runs fully offline and deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.detection.types import (
    AnomalyDetection,
    BoundingBox,
    FrameResult,
    SeverityLevel,
    UrgencyTag,
)


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """A deterministic 640x480 BGR image for tests."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def make_detection(
    class_name: str = "pothole",
    class_id: int = 0,
    confidence: float = 0.9,
    box: tuple[float, float, float, float] = (100, 100, 200, 200),
    depth_mm: float | None = 80.0,
) -> AnomalyDetection:
    """Factory producing a single :class:`AnomalyDetection`."""
    bbox = BoundingBox(*box)
    return AnomalyDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        area_px=bbox.area,
        depth_mm=depth_mm,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def sample_detection() -> AnomalyDetection:
    """A representative pothole detection."""
    return make_detection()


@pytest.fixture
def sample_frame_result(sample_detection: AnomalyDetection) -> FrameResult:
    """A frame result containing one detection."""
    return FrameResult(
        detections=[sample_detection],
        image_width=640,
        image_height=480,
        processing_time_ms=12.3,
        model_version="yolov11-test",
        road_condition_score=82.0,
    )


class FakeDetector:
    """A lightweight stand-in for :class:`AnomalyDetector`."""

    def __init__(self) -> None:
        self.is_loaded = True
        self.model_version = "yolov11-test"

    def load(self) -> None:
        """No-op load."""
        self.is_loaded = True

    def detect_image(self, image: np.ndarray) -> FrameResult:
        """Return a single fixed detection regardless of input."""
        h, w = image.shape[:2]
        det = make_detection(box=(10, 10, 110, 110))
        return FrameResult(
            detections=[det],
            image_width=w,
            image_height=h,
            processing_time_ms=5.0,
            model_version=self.model_version,
        )


@pytest.fixture
def fake_detector() -> FakeDetector:
    """Provide a fake detector instance."""
    return FakeDetector()


@pytest.fixture
def severity_levels() -> list[SeverityLevel]:
    """All severity levels for parametrised tests."""
    return [
        SeverityLevel.LOW,
        SeverityLevel.MEDIUM,
        SeverityLevel.HIGH,
        SeverityLevel.CRITICAL,
    ]


@pytest.fixture
def urgency_tags() -> list[UrgencyTag]:
    """All urgency tags for parametrised tests."""
    return [
        UrgencyTag.MONITOR,
        UrgencyTag.SCHEDULE_REPAIR,
        UrgencyTag.URGENT,
        UrgencyTag.IMMEDIATE,
    ]
