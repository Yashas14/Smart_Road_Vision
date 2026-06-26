"""Monocular depth estimation using MiDaS v3.1 (via ``torch.hub``).

Produces a relative inverse-depth map for a frame and extracts an approximate
pothole depth (in millimetres) at each anomaly centroid. Depth is a strong
signal for pothole severity. The estimator degrades gracefully when MiDaS or a
GPU is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.core.config import Settings, get_settings, load_yaml_config
from src.core.logging import get_logger
from src.detection.types import AnomalyDetection

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = get_logger(__name__)


class DepthEstimator:
    """MiDaS-based monocular depth estimator.

    Args:
        settings: Application settings; resolved from environment if omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = load_yaml_config("model_config.yaml").get("depth", {})
        self.enabled = bool(cfg.get("enabled", self.settings.enable_depth))
        self.model_type = cfg.get("model_type", self.settings.midas_model_type)
        self.hub_repo = cfg.get("hub_repo", "intel-isl/MiDaS")
        self.depth_scale_mm = float(cfg.get("depth_scale_mm", 120.0))
        self._model: Any | None = None
        self._transform: Any | None = None
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the MiDaS model and matching input transform."""
        if not self.enabled:
            logger.info("depth_disabled")
            return
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = torch.hub.load(self.hub_repo, self.model_type, trust_repo=True)
            self._model.to(self._device).eval()
            transforms = torch.hub.load(
                self.hub_repo, "transforms", trust_repo=True
            )
            if self.model_type == "MiDaS_small":
                self._transform = transforms.small_transform
            else:
                self._transform = transforms.dpt_transform
            logger.info("depth_loaded", model_type=self.model_type, device=self._device)
        except Exception as exc:  # pragma: no cover - heavy optional dep
            logger.warning("midas_load_failed", error=str(exc))
            self.enabled = False

    def estimate_depth_map(self, image: np.ndarray) -> np.ndarray | None:
        """Compute a normalised relative depth map for an image.

        Args:
            image: BGR image array.

        Returns:
            A float32 depth map normalised to ``[0, 1]`` (1 = nearest), or
            ``None`` if depth estimation is disabled/unavailable.
        """
        if not self.enabled or self._model is None or self._transform is None:
            return None
        try:
            import torch

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample = self._transform(rgb).to(self._device)
            with torch.no_grad():
                prediction = self._model(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy().astype(np.float32)
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max - d_min < 1e-6:
                return np.zeros_like(depth)
            return (depth - d_min) / (d_max - d_min)
        except Exception as exc:  # pragma: no cover
            logger.warning("depth_estimation_failed", error=str(exc))
            return None

    def annotate_depths(
        self, image: np.ndarray, detections: list[AnomalyDetection]
    ) -> list[AnomalyDetection]:
        """Estimate and attach a depth in millimetres to each detection.

        The local depression depth is approximated as the difference between the
        surrounding road surface (region median) and the anomaly centre.

        Args:
            image: BGR source image.
            detections: Detections to enrich (mutated in place).

        Returns:
            The same detections with ``depth_mm`` populated where possible.
        """
        if not detections:
            return detections
        depth_map = self.estimate_depth_map(image)
        if depth_map is None:
            return detections

        h, w = depth_map.shape[:2]
        for det in detections:
            cx, cy = det.bbox.centroid
            cx_i = int(np.clip(cx, 0, w - 1))
            cy_i = int(np.clip(cy, 0, h - 1))

            x1 = int(np.clip(det.bbox.x1, 0, w - 1))
            y1 = int(np.clip(det.bbox.y1, 0, h - 1))
            x2 = int(np.clip(det.bbox.x2, 1, w))
            y2 = int(np.clip(det.bbox.y2, 1, h))
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                continue

            surround_level = float(np.percentile(region, 25))
            centre_level = float(depth_map[cy_i, cx_i])
            # Larger inverse-depth at the surround vs. centre => a depression.
            relative_drop = max(0.0, surround_level - centre_level)
            det.depth_mm = round(relative_drop * self.depth_scale_mm * 10.0, 1)
        return detections
