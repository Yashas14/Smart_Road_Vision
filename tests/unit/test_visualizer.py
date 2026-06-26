"""Unit tests for :mod:`src.utils.visualizer`."""

from __future__ import annotations

import numpy as np

from src.utils.visualizer import (
    depth_heatmap,
    draw_summary_banner,
    overlay_depth,
    render_frame,
    severity_legend,
)
from tests.conftest import make_detection


def test_depth_heatmap_is_bgr_uint8() -> None:
    depth = np.linspace(0, 1, 100 * 100, dtype=np.float32).reshape(100, 100)
    heat = depth_heatmap(depth)
    assert heat.shape == (100, 100, 3)
    assert heat.dtype == np.uint8


def test_overlay_depth_resizes_mismatched_map(synthetic_image: np.ndarray) -> None:
    small_depth = np.zeros((50, 50), dtype=np.float32)
    out = overlay_depth(synthetic_image, small_depth, alpha=0.5)
    assert out.shape == synthetic_image.shape


def test_draw_summary_banner_keeps_shape(synthetic_image: np.ndarray) -> None:
    dets = [make_detection(), make_detection(class_name="hump")]
    out = draw_summary_banner(synthetic_image, dets, road_score=72.0)
    assert out.shape == synthetic_image.shape
    assert out is not synthetic_image


def test_draw_summary_banner_handles_no_detections(synthetic_image: np.ndarray) -> None:
    out = draw_summary_banner(synthetic_image, [], road_score=100.0)
    assert out.shape == synthetic_image.shape


def test_render_frame_with_and_without_banner(synthetic_image: np.ndarray) -> None:
    dets = [make_detection(box=(10, 10, 60, 60))]
    with_banner = render_frame(synthetic_image, dets, road_score=80.0, banner=True)
    without_banner = render_frame(synthetic_image, dets, road_score=80.0, banner=False)
    assert with_banner.shape == synthetic_image.shape
    assert without_banner.shape == synthetic_image.shape


def test_severity_legend_has_all_levels() -> None:
    legend = severity_legend()
    assert set(legend) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert all(v.startswith("#") for v in legend.values())
