"""Celery batch processing for folders and video files.

Exposes a configured Celery application plus tasks for processing a directory of
images and for processing a single video asynchronously. Designed to run inside
the ``celery_worker`` container defined in ``docker-compose.yml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from celery import Celery

from src.core.config import get_settings
from src.core.logging import get_logger
from src.pipeline.image_pipeline import ImagePipeline
from src.pipeline.video_pipeline import VideoPipeline

logger = get_logger(__name__)
settings = get_settings()

celery_app = Celery(
    "smartroadvision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=1800,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lazily constructed singletons within the worker process.
_image_pipeline: ImagePipeline | None = None


def _get_image_pipeline() -> ImagePipeline:
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = ImagePipeline(settings)
        _image_pipeline.warmup()
    return _image_pipeline


@celery_app.task(bind=True, name="batch.process_folder")
def process_folder_task(self: Any, folder: str) -> dict[str, Any]:
    """Process every image in a folder and return aggregate statistics.

    Args:
        folder: Path to a directory of images.

    Returns:
        A dict summarising per-file detection counts and totals.
    """
    pipeline = _get_image_pipeline()
    root = Path(folder)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    total = len(files)
    results: list[dict[str, Any]] = []
    total_detections = 0

    for i, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        result, _ = pipeline.process(image, annotate=False)
        total_detections += result.count
        results.append(
            {
                "file": path.name,
                "detections": result.count,
                "road_score": result.road_condition_score,
            }
        )
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": total, "file": path.name},
        )

    logger.info("batch_folder_done", files=total, detections=total_detections)
    return {
        "total_files": total,
        "total_detections": total_detections,
        "results": results,
    }


@celery_app.task(bind=True, name="batch.process_video")
def process_video_task(
    self: Any, source: str, output_path: str | None = None
) -> dict[str, Any]:
    """Process a video file asynchronously.

    Args:
        source: Path to the input video.
        output_path: Optional annotated output video path.

    Returns:
        A dict with the video processing summary.
    """
    pipeline = VideoPipeline(settings, pipeline=_get_image_pipeline())
    summary = pipeline.process(source, output_path=output_path)
    self.update_state(state="SUCCESS", meta={"frames": summary.total_frames})
    return {
        "total_frames": summary.total_frames,
        "total_detections": summary.total_detections,
        "unique_anomalies": summary.unique_anomalies,
        "avg_road_score": summary.avg_road_score,
        "per_class_counts": summary.per_class_counts,
        "output_path": summary.output_path,
    }
