"""Unit tests for :mod:`src.utils.image_utils`."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from src.core.exceptions import ValidationError
from src.utils.image_utils import (
    base64_to_image,
    bytes_to_image,
    image_to_base64,
    image_to_bytes,
    resize_keep_aspect,
)


def test_png_round_trip_is_lossless(synthetic_image: np.ndarray) -> None:
    encoded = image_to_bytes(synthetic_image, ext=".png")
    decoded = bytes_to_image(encoded)
    assert decoded.shape == synthetic_image.shape
    assert np.array_equal(decoded, synthetic_image)


def test_jpeg_round_trip_preserves_shape(synthetic_image: np.ndarray) -> None:
    encoded = image_to_bytes(synthetic_image, ext=".jpg", quality=85)
    decoded = bytes_to_image(encoded)
    assert decoded.shape == synthetic_image.shape


def test_base64_round_trip(synthetic_image: np.ndarray) -> None:
    b64 = image_to_base64(synthetic_image, ext=".png")
    # Must be valid base64.
    assert base64.b64decode(b64)
    decoded = base64_to_image(b64)
    assert np.array_equal(decoded, synthetic_image)


def test_base64_accepts_data_uri_prefix(synthetic_image: np.ndarray) -> None:
    b64 = image_to_base64(synthetic_image, ext=".png")
    data_uri = f"data:image/png;base64,{b64}"
    decoded = base64_to_image(data_uri)
    assert decoded.shape == synthetic_image.shape


def test_bytes_to_image_rejects_garbage() -> None:
    with pytest.raises(ValidationError):
        bytes_to_image(b"not-an-image")


def test_base64_to_image_rejects_invalid_base64() -> None:
    with pytest.raises(ValidationError):
        base64_to_image("%%%not-base64%%%")


def test_resize_keep_aspect_downscales_longest_side() -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    resized = resize_keep_aspect(image, max_side=400)
    assert max(resized.shape[:2]) == 400
    # Aspect ratio preserved (800:400 == 2:1).
    assert resized.shape[1] == 2 * resized.shape[0]


def test_resize_keep_aspect_noop_when_small(synthetic_image: np.ndarray) -> None:
    out = resize_keep_aspect(synthetic_image, max_side=5000)
    assert out.shape == synthetic_image.shape
