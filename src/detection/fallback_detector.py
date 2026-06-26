"""Classical computer-vision fallback detector (no PyTorch/YOLO required).

When the deep-learning stack (``ultralytics``/``torch``) or the trained weights
are unavailable, this detector keeps the whole pipeline functional by locating
road-surface anomalies with classical OpenCV techniques:

* **Potholes / degradation** — locally dark, compact blobs that are noticeably
  darker than the surrounding asphalt (illumination-normalised).
* **Cracks** — thin, elongated dark contours.

The output is fully compatible with the deep-learning path (same
:class:`~src.detection.types.AnomalyDetection` objects, including polygon masks
and a heuristic depth estimate), so severity scoring, annotation, reporting,
cost estimation and export all work end-to-end without any model download.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.core.logging import get_logger
from src.detection.postprocessor import non_max_suppression
from src.detection.types import AnomalyDetection, BoundingBox, FrameResult

logger = get_logger(__name__)

MODEL_VERSION = "opencv-heuristic-v1"


class ClassicalAnomalyDetector:
    """Heuristic OpenCV-based road anomaly detector.

    Args:
        class_map: Optional class-index to label mapping (for parity with YOLO).
        min_area_ratio: Minimum contour area as a fraction of the image area.
        max_area_ratio: Maximum contour area as a fraction of the image area.
        darkness_delta: How much darker than the local background a region must
            be (0-255) to be considered a candidate anomaly.
        max_detections: Hard cap on returned detections (highest confidence kept).
    """

    def __init__(
        self,
        class_map: dict[int, str] | None = None,
        min_area_ratio: float = 0.0006,
        max_area_ratio: float = 0.25,
        darkness_delta: float = 18.0,
        max_detections: int = 12,
    ) -> None:
        self.class_map = class_map or {0: "pothole", 1: "hump", 2: "crack", 3: "road_degradation"}
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.darkness_delta = darkness_delta
        self.max_detections = max_detections
        self.model_version = MODEL_VERSION

    # -- public API ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """Always ready; nothing to load."""
        return True

    def load(self) -> None:
        """No-op for API parity with the deep-learning detector."""

    def detect_image(self, image: np.ndarray) -> FrameResult:
        """Detect anomalies in a single BGR image.

        Args:
            image: HxWx3 BGR image (OpenCV convention).

        Returns:
            A :class:`FrameResult` with heuristic detections.
        """
        h, w = image.shape[:2]
        start = time.perf_counter()
        detections = self._detect(image, w, h)
        detections = non_max_suppression(detections, iou_threshold=0.4, class_agnostic=True)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[: self.max_detections]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "image_detected_fallback", count=len(detections), latency_ms=round(elapsed_ms, 2)
        )
        return FrameResult(
            detections=detections,
            image_width=w,
            image_height=h,
            processing_time_ms=elapsed_ms,
            model_version=self.model_version,
        )

    # -- internals ----------------------------------------------------------
    def _detect(self, image: np.ndarray, w: int, h: int) -> list[AnomalyDetection]:
        """Locate candidate anomalies via illumination-normalised thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Local background via a large-kernel blur; anomalies are darker than it.
        kernel = max(31, (min(w, h) // 12) | 1)  # odd kernel scaled to image size
        background = cv2.blur(gray, (kernel, kernel))
        diff = cv2.subtract(background, gray)  # bright where gray << background

        _, mask = cv2.threshold(diff, float(self.darkness_delta), 255, cv2.THRESH_BINARY)
        morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = float(w * h)
        min_area = self.min_area_ratio * image_area
        max_area = self.max_area_ratio * image_area

        detections: list[AnomalyDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 4 or bh < 4:
                continue
            rect_area = float(bw * bh)
            extent = area / rect_area if rect_area > 0 else 0.0
            aspect = bw / bh if bh > 0 else 0.0

            class_id, class_name = self._classify(aspect, extent, area, image_area)

            # Confidence from local contrast (darker => more confident).
            blob_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, 255, -1)
            contrast = float(cv2.mean(diff, mask=blob_mask)[0])
            confidence = float(np.clip(0.35 + contrast / 120.0, 0.30, 0.97))

            depth_mm = None
            if class_name in {"pothole", "road_degradation"}:
                depth_mm = round(float(np.clip(contrast / 80.0, 0.0, 1.0) * 150.0), 1)

            polygon = self._approx_polygon(contour)

            bbox = BoundingBox(float(x), float(y), float(x + bw), float(y + bh))
            detections.append(
                AnomalyDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(confidence, 3),
                    bbox=bbox,
                    polygon_mask=polygon,
                    depth_mm=depth_mm,
                    area_px=float(area),
                )
            )
        return detections

    def _classify(
        self, aspect: float, extent: float, area: float, image_area: float
    ) -> tuple[int, str]:
        """Map shape descriptors onto an anomaly class."""
        # Thin and elongated -> crack.
        if (aspect > 3.5 or aspect < 0.28) and extent < 0.55:
            return 2, "crack"
        # Compact and reasonably filled -> pothole.
        if extent >= 0.45 and area >= 0.0025 * image_area:
            return 0, "pothole"
        # Otherwise generic surface degradation.
        return 3, "road_degradation"

    @staticmethod
    def _approx_polygon(contour: np.ndarray) -> list[tuple[float, float]]:
        """Simplify a contour into a polygon mask for segmentation overlays."""
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [(float(p[0][0]), float(p[0][1])) for p in approx]
