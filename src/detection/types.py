"""Shared dataclasses and enums for the detection subsystem.

Kept in a dedicated module to avoid circular imports between the detector,
segmentor, depth estimator and severity scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SeverityLevel(StrEnum):
    """Discrete severity classification for an anomaly."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class UrgencyTag(StrEnum):
    """Maintenance urgency derived from severity."""

    MONITOR = "MONITOR"
    SCHEDULE_REPAIR = "SCHEDULE_REPAIR"
    URGENT = "URGENT"
    IMMEDIATE = "IMMEDIATE"


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned bounding box in absolute and normalised coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def xywhn(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Return normalised ``(cx, cy, w, h)`` relative to image dimensions."""
        cx, cy = self.centroid
        return (
            cx / img_w,
            cy / img_h,
            self.width / img_w,
            self.height / img_h,
        )


@dataclass(slots=True)
class AnomalyDetection:
    """Structured result for a single detected road anomaly.

    Attributes:
        class_id: Integer class index from the detector.
        class_name: Human-readable class label (e.g. ``"pothole"``).
        confidence: Detector confidence in ``[0, 1]``.
        bbox: Bounding box of the anomaly.
        polygon_mask: Optional segmentation polygon as ``[(x, y), ...]``.
        severity_score: Composite severity score in ``[0, 1]``.
        severity_level: Discrete :class:`SeverityLevel`.
        urgency: Maintenance :class:`UrgencyTag`.
        depth_mm: Estimated depth in millimetres (potholes), if available.
        area_px: Area in pixels.
        area_m2: Real-world area in m² when scale metadata is available.
        track_id: Persistent tracking id across video frames.
        timestamp: UTC capture/processing time.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    polygon_mask: list[tuple[float, float]] | None = None
    severity_score: float = 0.0
    severity_level: SeverityLevel = SeverityLevel.LOW
    urgency: UrgencyTag = UrgencyTag.MONITOR
    depth_mm: float | None = None
    area_px: float = 0.0
    area_m2: float | None = None
    track_id: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        data = asdict(self)
        data["bbox_xyxy"] = list(self.bbox.xyxy())
        data["timestamp"] = self.timestamp.isoformat()
        data["severity_level"] = str(self.severity_level)
        data["urgency"] = str(self.urgency)
        return data


@dataclass(slots=True)
class FrameResult:
    """Aggregated detection result for a single image/frame."""

    detections: list[AnomalyDetection]
    image_width: int
    image_height: int
    processing_time_ms: float
    model_version: str
    frame_index: int | None = None
    road_condition_score: float | None = None

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "frame_index": self.frame_index,
            "road_condition_score": self.road_condition_score,
        }
