"""Post-processing: NMS, coordinate normalisation and annotation drawing.

Although ultralytics applies NMS internally, this module provides an additional
class-aware NMS pass (useful when merging detections from TTA or multiple
models) plus helpers to normalise coordinates and render annotated frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.logging import get_logger
from src.detection.types import AnomalyDetection, SeverityLevel

logger = get_logger(__name__)

_SEVERITY_COLORS: dict[SeverityLevel, tuple[int, int, int]] = {
    SeverityLevel.LOW: (0, 200, 0),  # green
    SeverityLevel.MEDIUM: (0, 215, 255),  # yellow
    SeverityLevel.HIGH: (0, 140, 255),  # orange
    SeverityLevel.CRITICAL: (0, 0, 255),  # red
}


def _iou(a: AnomalyDetection, b: AnomalyDetection) -> float:
    """Intersection-over-union between two detections' boxes."""
    ax1, ay1, ax2, ay2 = a.bbox.xyxy()
    bx1, by1, bx2, by2 = b.bbox.xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.bbox.area + b.bbox.area - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(
    detections: list[AnomalyDetection],
    iou_threshold: float = 0.45,
    class_agnostic: bool = False,
) -> list[AnomalyDetection]:
    """Apply greedy NMS, keeping the highest-confidence boxes.

    Args:
        detections: Candidate detections.
        iou_threshold: Boxes overlapping above this IoU are suppressed.
        class_agnostic: If False, suppression is applied per class.

    Returns:
        The filtered list of detections.
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[AnomalyDetection] = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [
            d
            for d in ordered
            if (not class_agnostic and d.class_id != best.class_id) or _iou(best, d) < iou_threshold
        ]
    return keep


def normalize_coordinates(
    detections: list[AnomalyDetection], image_width: int, image_height: int
) -> list[dict[str, float]]:
    """Return normalised ``(cx, cy, w, h)`` tuples for each detection.

    Args:
        detections: Detections to normalise.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        A list of dicts with normalised box coordinates.
    """
    out = []
    for det in detections:
        cx, cy, w, h = det.bbox.xywhn(image_width, image_height)
        out.append({"cx": cx, "cy": cy, "w": w, "h": h})
    return out


def draw_annotations(
    image: np.ndarray,
    detections: list[AnomalyDetection],
    draw_masks: bool = True,
) -> np.ndarray:
    """Render bounding boxes, masks and severity labels onto a copy of the image.

    Args:
        image: BGR source image.
        detections: Detections to draw.
        draw_masks: Whether to overlay segmentation polygons.

    Returns:
        A new annotated BGR image.
    """
    canvas = image.copy()
    overlay = image.copy()

    for det in detections:
        color = _SEVERITY_COLORS.get(det.severity_level, (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in det.bbox.xyxy())

        if draw_masks and det.polygon_mask:
            pts = np.array(det.polygon_mask, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f} | {det.severity_level}"
        if det.depth_mm is not None:
            label += f" | {det.depth_mm:.0f}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if draw_masks:
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    return canvas
