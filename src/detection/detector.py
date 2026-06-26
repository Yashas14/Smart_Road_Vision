"""YOLOv11 anomaly detector wrapper.

Wraps the ``ultralytics`` YOLO model to provide three inference modes (single
image, video frame iterator, async stream), structured :class:`AnomalyDetection`
outputs, FP16 support on CUDA, and a warm-up routine executed at startup.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.config import Settings, get_settings, load_yaml_config
from src.core.exceptions import DetectionError
from src.core.logging import get_logger
from src.detection.fallback_detector import ClassicalAnomalyDetector
from src.detection.types import AnomalyDetection, BoundingBox, FrameResult

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from ultralytics import YOLO

logger = get_logger(__name__)

_DEFAULT_CLASS_MAP = {0: "pothole", 1: "hump", 2: "crack", 3: "road_degradation"}


def _resolve_device(requested: str) -> str:
    """Resolve the inference device, honouring ``auto``.

    Args:
        requested: ``auto``, ``cpu``, or an explicit CUDA device string.

    Returns:
        A concrete device string usable by torch/ultralytics.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present in prod
        return "cpu"


class AnomalyDetector:
    """High-level YOLOv11 detector for road surface anomalies.

    Args:
        settings: Application settings; resolved from environment if omitted.
        class_map: Optional override mapping of class index to label.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        class_map: dict[int, str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        cfg = load_yaml_config("model_config.yaml").get("yolo", {})
        self.class_map = class_map or cfg.get("classes") or _DEFAULT_CLASS_MAP
        self.imgsz = int(cfg.get("imgsz", self.settings.yolo_imgsz))
        self.confidence = float(cfg.get("confidence", self.settings.yolo_confidence))
        self.iou = float(cfg.get("iou", self.settings.yolo_iou))
        self.max_det = int(cfg.get("max_det", 300))
        self.augment = bool(cfg.get("augment", False))
        self.device = _resolve_device(self.settings.yolo_device)
        self.half = bool(self.settings.yolo_half) and self.device.startswith("cuda")
        self._model: YOLO | None = None
        self._fallback: ClassicalAnomalyDetector | None = None
        self.model_version = "yolov11-pothole-v2.0"

    @property
    def is_loaded(self) -> bool:
        """Whether a detector (deep-learning or fallback) is ready."""
        return self._model is not None or self._fallback is not None

    @property
    def using_fallback(self) -> bool:
        """Whether the classical OpenCV fallback is active."""
        return self._fallback is not None

    def load(self) -> None:
        """Load the YOLOv11 weights, falling back to classical CV if unavailable.

        If ``ultralytics``/``torch`` (or the weights) cannot be loaded, the
        detector transparently switches to a classical OpenCV heuristic so the
        rest of the pipeline keeps working without any model download.
        """
        weights = Path(self.settings.yolo_weights)
        try:
            from ultralytics import YOLO

            if not weights.exists():
                logger.warning(
                    "yolo_weights_missing_using_pretrained",
                    expected=str(weights),
                )
                # Fall back to a base YOLOv11 checkpoint that ultralytics fetches.
                self._model = YOLO("yolo11n.pt")
            else:
                self._model = YOLO(str(weights))
            self._model.to(self.device)
            self._fallback = None
            self._warmup()
            logger.info(
                "detector_loaded",
                device=self.device,
                half=self.half,
                weights=str(weights),
            )
        except Exception as exc:
            logger.warning(
                "yolo_unavailable_using_opencv_fallback",
                error=str(exc),
            )
            self._model = None
            self._fallback = ClassicalAnomalyDetector(class_map=self.class_map)
            self.model_version = self._fallback.model_version

    def _warmup(self) -> None:
        """Run a dummy inference so the first real request is fast."""
        if self._model is None:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self._model.predict(
                dummy, imgsz=self.imgsz, device=self.device, verbose=False
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("detector_warmup_failed", error=str(exc))

    def _ensure_loaded(self) -> YOLO | None:
        if self._model is None and self._fallback is None:
            self.load()
        return self._model

    def _parse_results(
        self, result: Any, img_w: int, img_h: int
    ) -> list[AnomalyDetection]:
        """Convert a single ultralytics result into anomaly detections."""
        detections: list[AnomalyDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss, strict=False):
            bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
            detections.append(
                AnomalyDetection(
                    class_id=int(cls_id),
                    class_name=self.class_map.get(int(cls_id), f"class_{cls_id}"),
                    confidence=float(conf),
                    bbox=bbox,
                    area_px=bbox.area,
                )
            )
        return detections

    def detect_image(self, image: np.ndarray) -> FrameResult:
        """Run detection on a single BGR image array.

        Args:
            image: HxWx3 BGR image (OpenCV convention).

        Returns:
            A :class:`FrameResult` with all detected anomalies.

        Raises:
            DetectionError: If inference fails.
        """
        model = self._ensure_loaded()
        if model is None and self._fallback is not None:
            return self._fallback.detect_image(image)
        h, w = image.shape[:2]
        start = time.perf_counter()
        try:
            results = model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_det,
                half=self.half,
                augment=self.augment,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectionError(f"YOLO inference failed: {exc}") from exc

        detections = self._parse_results(results[0], w, h) if results else []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "image_detected",
            count=len(detections),
            latency_ms=round(elapsed_ms, 2),
        )
        return FrameResult(
            detections=detections,
            image_width=w,
            image_height=h,
            processing_time_ms=elapsed_ms,
            model_version=self.model_version,
        )

    def detect_frames(
        self, frames: Iterable[np.ndarray]
    ) -> Iterator[FrameResult]:
        """Lazily run detection over an iterable of frames.

        Args:
            frames: Iterable of BGR image arrays.

        Yields:
            One :class:`FrameResult` per frame, with ``frame_index`` populated.
        """
        for idx, frame in enumerate(frames):
            result = self.detect_image(frame)
            result.frame_index = idx
            yield result

    async def detect_stream(
        self, frames: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[FrameResult]:
        """Asynchronously run detection over a stream of frames.

        Args:
            frames: Async iterator of BGR image arrays.

        Yields:
            One :class:`FrameResult` per frame.
        """
        idx = 0
        async for frame in frames:
            result = self.detect_image(frame)
            result.frame_index = idx
            idx += 1
            yield result
