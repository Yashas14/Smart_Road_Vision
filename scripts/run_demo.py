"""CLI demo: run the detection pipeline on an image, video or webcam.

Examples::

    python scripts/run_demo.py --source data/samples/road.jpg --save out.jpg
    python scripts/run_demo.py --source data/samples/drive.mp4 --save out.mp4
    python scripts/run_demo.py --source webcam
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2

from src.core.logging import configure_logging, get_logger
from src.pipeline.image_pipeline import ImagePipeline
from src.pipeline.stream_pipeline import StreamPipeline
from src.pipeline.video_pipeline import VideoPipeline

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _run_image(source: Path, save: str | None) -> None:
    """Run detection on a single image and optionally save the annotation."""
    pipeline = ImagePipeline()
    pipeline.warmup()
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Could not read image: {source}")
    result, annotated = pipeline.process(image, annotate=True)
    print(f"Detected {result.count} anomalies | road score {result.road_condition_score:.0f}/100")
    for det in result.detections:
        print(
            f"  - {det.class_name}: conf={det.confidence:.2f} "
            f"severity={det.severity_level} depth={det.depth_mm}mm"
        )
    if save and annotated is not None:
        cv2.imwrite(save, annotated)
        print(f"Saved annotated image to {save}")


def _run_video(source: Path, save: str | None) -> None:
    """Run detection on a video and optionally write an annotated copy."""
    pipeline = VideoPipeline()
    summary = pipeline.process(source, output_path=save)
    print(
        f"Processed {summary.total_frames} frames | "
        f"{summary.total_detections} detections | "
        f"{summary.unique_anomalies} unique anomalies | "
        f"avg road score {summary.avg_road_score:.0f}/100"
    )
    if save:
        print(f"Saved annotated video to {save}")


async def _run_stream(source: str) -> None:
    """Run a live stream demo, printing per-frame statistics."""
    pipeline = StreamPipeline()
    stop = asyncio.Event()
    try:
        async for frame in pipeline.stream(source, stop=stop):
            print(
                f"frame {frame.frame_index} | fps {frame.fps} | "
                f"anomalies {frame.result.count} | "
                f"alert {frame.density_alert}"
            )
            if frame.frame_index >= 300:  # cap demo length
                break
    finally:
        stop.set()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(description="SmartRoadVision demo runner")
    parser.add_argument("--source", required=True, help="image/video path, RTSP URL, or 'webcam'")
    parser.add_argument("--save", default=None, help="output path for annotated media")
    args = parser.parse_args()

    if args.source == "webcam" or args.source.startswith(("rtsp://", "http://", "https://")):
        asyncio.run(_run_stream(args.source))
        return

    path = Path(args.source)
    if not path.exists():
        raise SystemExit(f"Source not found: {path}")
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        _run_image(path, args.save)
    elif suffix in _VIDEO_EXTS:
        _run_video(path, args.save)
    else:
        raise SystemExit(f"Unsupported source type: {suffix}")


if __name__ == "__main__":
    main()
