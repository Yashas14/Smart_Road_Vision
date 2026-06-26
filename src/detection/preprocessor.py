"""Image preprocessing for robust detection in adverse conditions.

Implements CLAHE-based contrast enhancement, denoising, and optional night/rain
augmentation. These steps materially improve recall on low-light and wet-road
imagery captured by dashcams and drones.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.config import Settings, get_settings, load_yaml_config
from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Configurable preprocessing pipeline for road imagery.

    Args:
        settings: Application settings; resolved from environment if omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = load_yaml_config("model_config.yaml").get("preprocessing", {})
        self.clahe_clip = float(cfg.get("clahe_clip_limit", 2.0))
        grid = cfg.get("clahe_tile_grid", [8, 8])
        self.clahe_grid = (int(grid[0]), int(grid[1]))
        self.denoise_h = int(cfg.get("denoise_h", 7))
        self.enable_night = bool(cfg.get("enable_night_mode", True))
        self.enable_rain = bool(cfg.get("enable_rain_mode", False))
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of the LAB colour space.

        Args:
            image: BGR image array.

        Returns:
            Contrast-enhanced BGR image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = self._clahe.apply(lightness)
        merged = cv2.merge((lightness, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply fast non-local-means colour denoising.

        Args:
            image: BGR image array.

        Returns:
            Denoised BGR image.
        """
        return cv2.fastNlMeansDenoisingColored(image, None, self.denoise_h, self.denoise_h, 7, 21)

    @staticmethod
    def _is_low_light(image: np.ndarray, threshold: float = 80.0) -> bool:
        """Heuristically detect low-light frames by mean luminance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < threshold

    def enhance_night(self, image: np.ndarray) -> np.ndarray:
        """Brighten and gamma-correct a low-light image.

        Args:
            image: BGR image array.

        Returns:
            Brightened BGR image.
        """
        gamma = 1.6
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    def remove_rain(self, image: np.ndarray) -> np.ndarray:
        """Reduce rain streaks via guided morphological filtering.

        Args:
            image: BGR image array.

        Returns:
            De-rained BGR image.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.bilateralFilter(opened, 5, 50, 50)

    def process(
        self,
        image: np.ndarray,
        denoise: bool = False,
        auto_night: bool = True,
    ) -> np.ndarray:
        """Run the configured preprocessing pipeline.

        Args:
            image: BGR image array.
            denoise: Force denoising regardless of conditions.
            auto_night: Auto-apply night enhancement on low-light frames.

        Returns:
            The preprocessed BGR image.

        Raises:
            PreprocessingError: If the input is empty or invalid.
        """
        if image is None or image.size == 0:
            raise PreprocessingError("Empty image passed to preprocessor")

        out = image
        if auto_night and self.enable_night and self._is_low_light(out):
            out = self.enhance_night(out)
            logger.debug("night_enhancement_applied")
        out = self.apply_clahe(out)
        if self.enable_rain:
            out = self.remove_rain(out)
        if denoise:
            out = self.denoise(out)
        return out
