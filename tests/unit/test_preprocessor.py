"""Unit tests for the preprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.exceptions import PreprocessingError
from src.detection.preprocessor import Preprocessor


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor()


def test_process_returns_same_shape(
    preprocessor: Preprocessor, synthetic_image: np.ndarray
) -> None:
    out = preprocessor.process(synthetic_image)
    assert out.shape == synthetic_image.shape
    assert out.dtype == np.uint8


def test_clahe_preserves_shape(
    preprocessor: Preprocessor, synthetic_image: np.ndarray
) -> None:
    out = preprocessor.apply_clahe(synthetic_image)
    assert out.shape == synthetic_image.shape


def test_low_light_detection() -> None:
    dark = np.full((100, 100, 3), 10, dtype=np.uint8)
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert Preprocessor._is_low_light(dark) is True
    assert Preprocessor._is_low_light(bright) is False


def test_night_enhancement_brightens(preprocessor: Preprocessor) -> None:
    dark = np.full((50, 50, 3), 20, dtype=np.uint8)
    enhanced = preprocessor.enhance_night(dark)
    assert float(enhanced.mean()) > float(dark.mean())


def test_empty_image_raises(preprocessor: Preprocessor) -> None:
    with pytest.raises(PreprocessingError):
        preprocessor.process(np.empty((0, 0, 3), dtype=np.uint8))


def test_denoise_keeps_shape(
    preprocessor: Preprocessor, synthetic_image: np.ndarray
) -> None:
    out = preprocessor.process(synthetic_image, denoise=True)
    assert out.shape == synthetic_image.shape
