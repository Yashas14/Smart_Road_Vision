"""Integration tests for the video pipeline (synthetic video, fake detector)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detection.types import FrameResult
from src.pipeline.video_pipeline import VideoPipeline, _IoUTracker
from tests.conftest import make_detection


class _FakeImagePipeline:
    """Image pipeline stub returning one detection per frame."""

    def process(self, image: np.ndarray, denoise: bool = False, annotate: bool = True):
        h, w = image.shape[:2]
        det = make_detection(box=(10, 10, 60, 60))
        result = FrameResult(
            detections=[det],
            image_width=w,
            image_height=h,
            processing_time_ms=1.0,
            model_version="yolov11-test",
            road_condition_score=90.0,
        )
        return result, image


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a short synthetic MP4 for testing."""
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.full((120, 160, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_iou_tracker_assigns_stable_ids() -> None:
    tracker = _IoUTracker(iou_threshold=0.3)
    det1 = make_detection(box=(0, 0, 50, 50))
    tracker.update([det1])
    first_id = det1.track_id

    det2 = make_detection(box=(2, 2, 52, 52))  # overlapping next frame
    tracker.update([det2])
    assert det2.track_id == first_id


def test_iou_tracker_new_id_for_distinct() -> None:
    tracker = _IoUTracker(iou_threshold=0.3)
    a = make_detection(box=(0, 0, 30, 30))
    b = make_detection(box=(200, 200, 230, 230))
    tracker.update([a])
    tracker.update([b])
    assert a.track_id != b.track_id


def test_video_pipeline_processes_all_frames(synthetic_video: Path) -> None:
    pipeline = VideoPipeline(pipeline=_FakeImagePipeline())
    summary = pipeline.process(synthetic_video)
    assert summary.total_frames == 10
    assert summary.total_detections == 10
    assert summary.unique_anomalies >= 1
    assert "pothole" in summary.per_class_counts


def test_video_pipeline_writes_output(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "annotated.mp4"
    pipeline = VideoPipeline(pipeline=_FakeImagePipeline())
    summary = pipeline.process(synthetic_video, output_path=out)
    assert out.exists()
    assert summary.output_path == str(out)


def test_iter_results_yields_frame_indices(synthetic_video: Path) -> None:
    pipeline = VideoPipeline(pipeline=_FakeImagePipeline())
    results = list(pipeline.iter_results(synthetic_video))
    assert len(results) == 10
    assert results[0].frame_index == 0
