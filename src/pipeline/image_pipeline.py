"""Single-image detection pipeline.

Orchestrates the full per-image flow:
``preprocess -> detect -> segment -> depth -> score -> annotate``.
The pipeline lazily constructs and caches its component models so it can be
reused across many requests without reloading weights.
"""

from __future__ import annotations

import numpy as np

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.detection.depth_estimator import DepthEstimator
from src.detection.detector import AnomalyDetector
from src.detection.postprocessor import draw_annotations
from src.detection.preprocessor import Preprocessor
from src.detection.segmentor import AnomalySegmentor
from src.detection.severity_scorer import SeverityScorer
from src.detection.types import FrameResult

logger = get_logger(__name__)


class ImagePipeline:
    """End-to-end pipeline for processing a single image.

    Args:
        settings: Application settings; resolved from environment if omitted.
        detector: Optional pre-built detector (enables sharing across pipelines).
        segmentor: Optional pre-built segmentor.
        depth_estimator: Optional pre-built depth estimator.
        scorer: Optional pre-built severity scorer.
        preprocessor: Optional pre-built preprocessor.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        detector: AnomalyDetector | None = None,
        segmentor: AnomalySegmentor | None = None,
        depth_estimator: DepthEstimator | None = None,
        scorer: SeverityScorer | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.detector = detector or AnomalyDetector(self.settings)
        self.segmentor = segmentor or AnomalySegmentor(self.settings)
        self.depth_estimator = depth_estimator or DepthEstimator(self.settings)
        self.scorer = scorer or SeverityScorer(self.settings)
        self.preprocessor = preprocessor or Preprocessor(self.settings)
        self._warmed = False

    def warmup(self) -> None:
        """Eagerly load every component model."""
        self.detector.load()
        self.segmentor.load()
        self.depth_estimator.load()
        self._warmed = True
        logger.info("image_pipeline_ready")

    def process(
        self,
        image: np.ndarray,
        denoise: bool = False,
        annotate: bool = True,
    ) -> tuple[FrameResult, np.ndarray | None]:
        """Process a single BGR image end-to-end.

        Args:
            image: BGR image array.
            denoise: Apply denoising during preprocessing.
            annotate: Produce an annotated output image.

        Returns:
            A tuple of the :class:`FrameResult` and the annotated image (or
            ``None`` when ``annotate`` is False).
        """
        if not self._warmed:
            self.warmup()

        processed = self.preprocessor.process(image, denoise=denoise)
        result = self.detector.detect_image(processed)

        if result.detections:
            self.segmentor.segment(processed, result.detections)
            self.depth_estimator.annotate_depths(processed, result.detections)
            self.scorer.score(result.detections, result.image_width, result.image_height)

        result.road_condition_score = self.scorer.road_condition_score(result.detections)

        annotated = None
        if annotate:
            annotated = draw_annotations(image, result.detections)

        logger.info(
            "image_pipeline_done",
            count=result.count,
            road_score=result.road_condition_score,
            latency_ms=round(result.processing_time_ms, 2),
        )
        return result, annotated
