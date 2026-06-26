"""Detection subsystem: detector, segmentor, depth, severity, pre/post-processing."""

from src.detection.types import (
    AnomalyDetection,
    BoundingBox,
    FrameResult,
    SeverityLevel,
    UrgencyTag,
)

__all__ = [
    "AnomalyDetection",
    "BoundingBox",
    "FrameResult",
    "SeverityLevel",
    "UrgencyTag",
]
