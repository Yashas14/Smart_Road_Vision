"""Frame-by-frame video detection pipeline with optional tracking.

Reads a video with OpenCV, runs the image pipeline on each frame, assigns
persistent track ids (ByteTrack via ``boxmot`` when available, otherwise a
lightweight IoU tracker fallback) and optionally writes an annotated output
video.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.core.config import Settings, get_settings
from src.core.exceptions import StreamError
from src.core.logging import get_logger
from src.detection.postprocessor import draw_annotations
from src.detection.types import AnomalyDetection, FrameResult
from src.pipeline.image_pipeline import ImagePipeline

logger = get_logger(__name__)


class _IoUTracker:
    """Minimal IoU-based tracker used when ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: list[dict] = []

    @staticmethod
    def _iou(a: AnomalyDetection, box: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a.bbox.xyxy()
        bx1, by1, bx2, by2 = box
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = a.bbox.area + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[AnomalyDetection]) -> None:
        """Assign ``track_id`` to each detection (mutated in place)."""
        for track in self._tracks:
            track["age"] += 1
        for det in detections:
            best, best_iou = None, self.iou_threshold
            for track in self._tracks:
                score = self._iou(det, track["box"])
                if score >= best_iou:
                    best, best_iou = track, score
            if best is not None:
                best["box"] = det.bbox.xyxy()
                best["age"] = 0
                det.track_id = best["id"]
            else:
                det.track_id = self._next_id
                self._tracks.append({"id": self._next_id, "box": det.bbox.xyxy(), "age": 0})
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]


@dataclass(slots=True)
class VideoSummary:
    """Aggregate statistics for a processed video."""

    total_frames: int = 0
    total_detections: int = 0
    unique_anomalies: int = 0
    output_path: str | None = None
    avg_road_score: float = 100.0
    per_class_counts: dict[str, int] = field(default_factory=dict)


class VideoPipeline:
    """Process a video file frame-by-frame.

    Args:
        settings: Application settings; resolved from environment if omitted.
        pipeline: Optional shared :class:`ImagePipeline`.
        sample_every: Process every Nth frame (1 = every frame).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: ImagePipeline | None = None,
        sample_every: int = 1,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or ImagePipeline(self.settings)
        self.sample_every = max(1, sample_every)
        self.tracker = self._build_tracker()

    def _build_tracker(self):
        """Build a ByteTrack tracker if boxmot is installed, else IoU fallback."""
        try:
            from boxmot import ByteTrack  # type: ignore

            logger.info("using_bytetrack_tracker")
            return ByteTrack()
        except Exception:
            logger.info("using_iou_tracker_fallback")
            return _IoUTracker()

    def _apply_tracking(self, detections: list[AnomalyDetection], frame: np.ndarray) -> None:
        """Assign track ids using the active tracker."""
        if isinstance(self.tracker, _IoUTracker):
            self.tracker.update(detections)
            return
        if not detections:
            return
        try:
            dets = np.array(
                [[*d.bbox.xyxy(), d.confidence, d.class_id] for d in detections],
                dtype=np.float32,
            )
            tracks = self.tracker.update(dets, frame)
            for det, track in zip(detections, tracks, strict=False):
                det.track_id = int(track[4])
        except Exception as exc:  # pragma: no cover
            logger.warning("tracking_failed", error=str(exc))

    def iter_results(self, source: str | Path) -> Iterator[FrameResult]:
        """Yield a :class:`FrameResult` for each processed frame.

        Args:
            source: Path to a video file.

        Yields:
            One :class:`FrameResult` per processed frame.

        Raises:
            StreamError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise StreamError(f"Cannot open video source: {source}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.sample_every == 0:
                    result, _ = self.pipeline.process(frame, annotate=False)
                    self._apply_tracking(result.detections, frame)
                    result.frame_index = idx
                    yield result
                idx += 1
        finally:
            cap.release()

    def process(self, source: str | Path, output_path: str | Path | None = None) -> VideoSummary:
        """Process a full video and optionally write an annotated copy.

        Args:
            source: Path to the input video.
            output_path: Optional path for the annotated output video.

        Returns:
            A :class:`VideoSummary` with aggregate statistics.

        Raises:
            StreamError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise StreamError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        summary = VideoSummary(output_path=str(output_path) if output_path else None)
        track_ids: set[int] = set()
        road_scores: list[float] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.sample_every == 0:
                    result, _ = self.pipeline.process(frame, annotate=False)
                    self._apply_tracking(result.detections, frame)
                    summary.total_detections += result.count
                    for det in result.detections:
                        if det.track_id is not None:
                            track_ids.add(det.track_id)
                        summary.per_class_counts[det.class_name] = (
                            summary.per_class_counts.get(det.class_name, 0) + 1
                        )
                    if result.road_condition_score is not None:
                        road_scores.append(result.road_condition_score)
                    if writer is not None:
                        writer.write(draw_annotations(frame, result.detections))
                elif writer is not None:
                    writer.write(frame)
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        summary.total_frames = idx
        summary.unique_anomalies = len(track_ids)
        summary.avg_road_score = (
            round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0
        )
        logger.info(
            "video_processed",
            frames=summary.total_frames,
            detections=summary.total_detections,
            unique=summary.unique_anomalies,
        )
        return summary
