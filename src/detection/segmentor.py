"""SAM2 (Segment Anything Model 2) segmentation wrapper.

Refines coarse YOLO bounding boxes into precise instance polygons. The wrapper
degrades gracefully: if SAM2 weights are unavailable, it falls back to deriving
a rectangular polygon from the bounding box so downstream code keeps working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import Settings, get_settings, load_yaml_config
from src.core.logging import get_logger
from src.detection.types import AnomalyDetection

logger = get_logger(__name__)


def _mask_to_polygon(mask: np.ndarray, max_points: int = 60) -> list[tuple[float, float]]:
    """Convert a binary mask into a simplified polygon contour.

    Args:
        mask: 2D boolean/uint8 mask.
        max_points: Approximate upper bound on returned vertices.

    Returns:
        Polygon as a list of ``(x, y)`` vertices (empty if no contour found).
    """
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        step = len(pts) // max_points
        pts = pts[::step]
    return [(float(x), float(y)) for x, y in pts]


class AnomalySegmentor:
    """Wrapper around SAM2 for instance segmentation of detected anomalies.

    Args:
        settings: Application settings; resolved from environment if omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = load_yaml_config("model_config.yaml").get("segmentation", {})
        self.enabled = bool(cfg.get("enabled", self.settings.enable_segmentation))
        self.checkpoint = Path(cfg.get("checkpoint", self.settings.sam2_checkpoint))
        self.config = cfg.get("config", self.settings.sam2_config)
        self._predictor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        """Attempt to load SAM2; silently disables on failure."""
        if not self.enabled:
            logger.info("segmentor_disabled")
            return
        if not self.checkpoint.exists():
            logger.warning("sam2_checkpoint_missing", path=str(self.checkpoint))
            self.enabled = False
            return
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            model = build_sam2(self.config, str(self.checkpoint))
            self._predictor = SAM2ImagePredictor(model)
            logger.info("segmentor_loaded", checkpoint=str(self.checkpoint))
        except Exception as exc:  # pragma: no cover - optional heavy dep
            logger.warning("sam2_load_failed_fallback_bbox", error=str(exc))
            self.enabled = False

    def _bbox_polygon(self, det: AnomalyDetection) -> list[tuple[float, float]]:
        """Rectangular fallback polygon derived from the bounding box."""
        x1, y1, x2, y2 = det.bbox.xyxy()
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def segment(
        self, image: np.ndarray, detections: list[AnomalyDetection]
    ) -> list[AnomalyDetection]:
        """Attach polygon masks to each detection.

        Args:
            image: BGR source image.
            detections: Detections to enrich with polygons (mutated in place).

        Returns:
            The same list with ``polygon_mask`` populated on each detection.
        """
        if not detections:
            return detections

        if not self.enabled or self._predictor is None:
            for det in detections:
                det.polygon_mask = self._bbox_polygon(det)
            return detections

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)
            for det in detections:
                box = np.array(det.bbox.xyxy(), dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box[None, :], multimask_output=False
                )
                mask = masks[0]
                polygon = _mask_to_polygon(mask)
                det.polygon_mask = polygon or self._bbox_polygon(det)
                det.area_px = float(mask.sum()) or det.area_px
        except Exception as exc:  # pragma: no cover
            logger.warning("segmentation_failed_fallback", error=str(exc))
            for det in detections:
                if det.polygon_mask is None:
                    det.polygon_mask = self._bbox_polygon(det)
        return detections
