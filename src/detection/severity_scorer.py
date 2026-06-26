"""Composite severity scoring for road anomalies.

Combines four normalised signals — relative bounding-box area, estimated depth,
detector confidence and a per-class severity weight — into a single score in
``[0, 1]``, then maps that score onto discrete severity levels and maintenance
urgency tags.
"""

from __future__ import annotations

from src.core.config import Settings, get_settings, load_yaml_config
from src.core.logging import get_logger
from src.detection.types import (
    AnomalyDetection,
    SeverityLevel,
    UrgencyTag,
)

logger = get_logger(__name__)

_DEFAULT_CLASS_WEIGHTS = {
    "pothole": 1.0,
    "road_degradation": 0.8,
    "crack": 0.6,
    "hump": 0.5,
}

_SEVERITY_TO_URGENCY = {
    SeverityLevel.LOW: UrgencyTag.MONITOR,
    SeverityLevel.MEDIUM: UrgencyTag.SCHEDULE_REPAIR,
    SeverityLevel.HIGH: UrgencyTag.URGENT,
    SeverityLevel.CRITICAL: UrgencyTag.IMMEDIATE,
}


class SeverityScorer:
    """Rule-based composite severity scorer.

    The score is::

        severity = w_area*area_norm + w_depth*depth_norm
                 + w_conf*confidence + w_class*class_weight

    Args:
        settings: Application settings; resolved from environment if omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = load_yaml_config("model_config.yaml").get("severity", {})
        weights = cfg.get("weights") or self.settings.severity_weights
        self.w_area = float(weights.get("area", 0.30))
        self.w_depth = float(weights.get("depth", 0.35))
        self.w_conf = float(weights.get("confidence", 0.15))
        self.w_class = float(weights.get("class", 0.20))
        self.class_weights = cfg.get("class_weights") or _DEFAULT_CLASS_WEIGHTS
        thresholds = cfg.get("thresholds", {})
        self.t_low = float(thresholds.get("low", 0.30))
        self.t_medium = float(thresholds.get("medium", 0.60))
        self.t_high = float(thresholds.get("high", 0.80))
        self.depth_reference_mm = float(cfg.get("depth_reference_mm", 150.0))

    def _classify(self, score: float) -> SeverityLevel:
        """Map a continuous score onto a discrete severity level."""
        if score < self.t_low:
            return SeverityLevel.LOW
        if score < self.t_medium:
            return SeverityLevel.MEDIUM
        if score < self.t_high:
            return SeverityLevel.HIGH
        return SeverityLevel.CRITICAL

    def score_one(
        self, det: AnomalyDetection, image_width: int, image_height: int
    ) -> AnomalyDetection:
        """Compute and assign the severity for a single detection.

        Args:
            det: Detection to score (mutated in place).
            image_width: Source image width in pixels.
            image_height: Source image height in pixels.

        Returns:
            The detection with severity fields populated.
        """
        image_area = max(1.0, float(image_width * image_height))
        area_norm = min(1.0, det.bbox.area / image_area * 8.0)

        if det.depth_mm is not None:
            depth_norm = min(1.0, det.depth_mm / self.depth_reference_mm)
        else:
            depth_norm = 0.0

        confidence = max(0.0, min(1.0, det.confidence))
        class_weight = float(self.class_weights.get(det.class_name, 0.5))

        score = (
            self.w_area * area_norm
            + self.w_depth * depth_norm
            + self.w_conf * confidence
            + self.w_class * class_weight
        )
        score = round(max(0.0, min(1.0, score)), 4)

        det.severity_score = score
        det.severity_level = self._classify(score)
        det.urgency = _SEVERITY_TO_URGENCY[det.severity_level]
        return det

    def score(
        self,
        detections: list[AnomalyDetection],
        image_width: int,
        image_height: int,
    ) -> list[AnomalyDetection]:
        """Score every detection in a frame.

        Args:
            detections: Detections to score (mutated in place).
            image_width: Source image width in pixels.
            image_height: Source image height in pixels.

        Returns:
            The scored detections.
        """
        for det in detections:
            self.score_one(det, image_width, image_height)
        return detections

    @staticmethod
    def road_condition_score(detections: list[AnomalyDetection]) -> float:
        """Compute a 0-100 road condition score (100 = perfect).

        Each anomaly subtracts a penalty proportional to its severity.

        Args:
            detections: Detections present in the frame/segment.

        Returns:
            Road condition score clamped to ``[0, 100]``.
        """
        if not detections:
            return 100.0
        penalty = sum(8.0 + 17.0 * d.severity_score for d in detections)
        return round(max(0.0, 100.0 - penalty), 1)
