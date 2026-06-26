"""Real-time RTSP/webcam stream processing pipeline.

Provides an async generator that reads frames from a camera or RTSP source in a
background thread (so blocking OpenCV reads never stall the event loop), runs
detection, maintains a sliding-window anomaly-density alert, and yields
annotated JPEG bytes suitable for WebSocket broadcast.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import cv2
import numpy as np

from src.core.config import Settings, get_settings
from src.core.exceptions import StreamError
from src.core.logging import get_logger
from src.detection.types import FrameResult
from src.pipeline.image_pipeline import ImagePipeline

logger = get_logger(__name__)


@dataclass(slots=True)
class StreamFrame:
    """A single processed stream frame ready for transport."""

    frame_index: int
    result: FrameResult
    jpeg_bytes: bytes
    fps: float
    density_alert: bool


def _resolve_source(source: str) -> int | str:
    """Map ``"webcam"``/``"0"`` to a device index, else return the URL."""
    if source in {"webcam", "0"}:
        return 0
    if source.isdigit():
        return int(source)
    return source


class StreamPipeline:
    """Async real-time stream processor.

    Args:
        settings: Application settings; resolved from environment if omitted.
        pipeline: Optional shared :class:`ImagePipeline`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: ImagePipeline | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or ImagePipeline(self.settings)
        self.max_fps = self.settings.stream_max_fps
        self.density_window = self.settings.stream_density_window
        self.density_threshold = self.settings.stream_density_alert_threshold
        self.jpeg_quality = 80

    async def _read_frames(
        self, source: int | str, stop: asyncio.Event
    ) -> AsyncIterator[np.ndarray]:
        """Yield frames from the capture device without blocking the loop."""
        cap = await asyncio.to_thread(cv2.VideoCapture, source)
        if not cap.isOpened():
            raise StreamError(f"Cannot open stream source: {source}")
        try:
            while not stop.is_set():
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    logger.warning("stream_read_failed", source=str(source))
                    break
                yield frame
        finally:
            await asyncio.to_thread(cap.release)

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        """Encode a BGR image as JPEG bytes."""
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", image, params)
        if not ok:
            raise StreamError("JPEG encoding failed")
        return buf.tobytes()

    async def stream(
        self, source: str, stop: asyncio.Event | None = None
    ) -> AsyncIterator[StreamFrame]:
        """Process a live stream and yield annotated frames.

        Args:
            source: ``"webcam"``, a device index, or an RTSP/HTTP URL.
            stop: Optional event to request graceful shutdown.

        Yields:
            :class:`StreamFrame` objects with annotated JPEG bytes and metadata.

        Raises:
            StreamError: If the source cannot be opened.
        """
        stop = stop or asyncio.Event()
        resolved = _resolve_source(source)
        min_interval = 1.0 / max(1, self.max_fps)
        density: deque[int] = deque(maxlen=self.density_window)

        idx = 0
        last_emit = 0.0
        last_ts = time.perf_counter()

        async for frame in self._read_frames(resolved, stop):
            now = time.perf_counter()
            if now - last_emit < min_interval:
                continue  # throttle to max_fps

            result, annotated = self.pipeline.process(frame, annotate=True)
            result.frame_index = idx
            density.append(result.count)
            total_recent = sum(density)
            alert = total_recent >= self.density_threshold

            dt = now - last_ts
            fps = (1.0 / dt) if dt > 0 else 0.0
            last_ts = now
            last_emit = now

            jpeg = self._encode_jpeg(annotated if annotated is not None else frame)

            if alert:
                logger.warning(
                    "anomaly_density_alert",
                    window=self.density_window,
                    recent=total_recent,
                )

            yield StreamFrame(
                frame_index=idx,
                result=result,
                jpeg_bytes=jpeg,
                fps=round(fps, 1),
                density_alert=alert,
            )
            idx += 1
