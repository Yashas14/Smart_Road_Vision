"""Image utilities: base64 encode/decode, resizing and format conversion."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from src.core.exceptions import ValidationError


def bytes_to_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array.

    Args:
        data: Raw encoded image bytes (PNG/JPEG/etc.).

    Returns:
        Decoded HxWx3 BGR image.

    Raises:
        ValidationError: If the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError("Uploaded file is not a valid image")
    return image


def image_to_bytes(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> bytes:
    """Encode a BGR image to bytes.

    Args:
        image: BGR image array.
        ext: Output extension (``.jpg`` or ``.png``).
        quality: JPEG quality (ignored for PNG).

    Returns:
        Encoded image bytes.

    Raises:
        ValidationError: If encoding fails.
    """
    params: list[int] = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValidationError(f"Failed to encode image as {ext}")
    return buf.tobytes()


def image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR image as a base64 data string (no data-URI prefix).

    Args:
        image: BGR image array.
        ext: Output extension.

    Returns:
        Base64-encoded image string.
    """
    return base64.b64encode(image_to_bytes(image, ext)).decode("ascii")


def base64_to_image(b64: str) -> np.ndarray:
    """Decode a base64 image string into a BGR numpy array.

    Args:
        b64: Base64 string, optionally with a ``data:image/...;base64,`` prefix.

    Returns:
        Decoded BGR image.

    Raises:
        ValidationError: If decoding fails.
    """
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise ValidationError(f"Invalid base64 image data: {exc}") from exc
    return bytes_to_image(raw)


def resize_keep_aspect(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize so the longest side equals ``max_side``, preserving aspect ratio.

    Args:
        image: BGR image array.
        max_side: Target length of the longest edge.

    Returns:
        Resized image (original returned if already small enough).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
