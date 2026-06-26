"""OpenCV visualisation helpers: boxes, masks, depth overlays and dashboards.

Complements :mod:`src.detection.postprocessor` with richer composites used by
the dashboard and PDF report gallery (e.g. side-by-side depth heatmaps and a
compact summary banner).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.postprocessor import draw_annotations
from src.detection.types import AnomalyDetection, SeverityLevel

_SEVERITY_COLORS: dict[SeverityLevel, tuple[int, int, int]] = {
    SeverityLevel.LOW: (0, 200, 0),
    SeverityLevel.MEDIUM: (0, 215, 255),
    SeverityLevel.HIGH: (0, 140, 255),
    SeverityLevel.CRITICAL: (0, 0, 255),
}


def depth_heatmap(depth_map: np.ndarray) -> np.ndarray:
    """Colourise a normalised depth map for display.

    Args:
        depth_map: Float depth map in ``[0, 1]``.

    Returns:
        A BGR heatmap image.
    """
    normalized = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def overlay_depth(image: np.ndarray, depth_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a depth heatmap over the source image.

    Args:
        image: BGR source image.
        depth_map: Float depth map in ``[0, 1]`` matching the image size.
        alpha: Heatmap opacity in ``[0, 1]``.

    Returns:
        Blended BGR image.
    """
    heat = depth_heatmap(depth_map)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(heat, alpha, image, 1.0 - alpha, 0)


def draw_summary_banner(
    image: np.ndarray, detections: list[AnomalyDetection], road_score: float
) -> np.ndarray:
    """Draw a translucent summary banner along the top of the frame.

    Args:
        image: BGR image (already annotated, ideally).
        detections: Detections in the frame.
        road_score: Road condition score (0-100).

    Returns:
        A new image with a summary banner.
    """
    out = image.copy()
    h, w = out.shape[:2]
    banner_h = 40
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    summary = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "no anomalies"
    text = f"Anomalies: {len(detections)} ({summary})  |  Road score: {road_score:.0f}/100"
    cv2.putText(
        out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def render_frame(
    image: np.ndarray,
    detections: list[AnomalyDetection],
    road_score: float = 100.0,
    banner: bool = True,
) -> np.ndarray:
    """Produce a fully annotated frame with boxes, masks and a summary banner.

    Args:
        image: BGR source image.
        detections: Detections to render.
        road_score: Road condition score for the banner.
        banner: Whether to draw the summary banner.

    Returns:
        The rendered BGR image.
    """
    annotated = draw_annotations(image, detections)
    if banner:
        annotated = draw_summary_banner(annotated, detections, road_score)
    return annotated


def severity_legend() -> dict[str, str]:
    """Return a colour legend mapping severity to a hex colour (for UI)."""
    return {
        "LOW": "#00c800",
        "MEDIUM": "#ffd700",
        "HIGH": "#ff8c00",
        "CRITICAL": "#ff0000",
    }
