"""Geospatial utilities: GPS EXIF extraction and GeoJSON generation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class GeoPoint:
    """A WGS84 geographic coordinate."""

    latitude: float
    longitude: float
    altitude: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }


def _ratio_to_float(value: Any) -> float:
    """Convert a piexif rational ``(num, den)`` to float."""
    try:
        num, den = value
        return float(num) / float(den) if den else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return float(value)


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert degrees/minutes/seconds rationals + hemisphere ref to decimal.

    Args:
        dms: Tuple of three rationals (degrees, minutes, seconds).
        ref: Hemisphere reference (``N``/``S``/``E``/``W``).

    Returns:
        Decimal degrees, signed by hemisphere.
    """
    degrees = _ratio_to_float(dms[0])
    minutes = _ratio_to_float(dms[1])
    seconds = _ratio_to_float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in {"S", "W"}:
        decimal = -decimal
    return round(decimal, 7)


def extract_gps_from_exif(image_bytes: bytes) -> GeoPoint | None:
    """Extract GPS coordinates from image EXIF metadata.

    Args:
        image_bytes: Raw image bytes (JPEG/TIFF carry EXIF).

    Returns:
        A :class:`GeoPoint`, or ``None`` if no GPS metadata is present.
    """
    try:
        import piexif

        exif = piexif.load(image_bytes)
        gps = exif.get("GPS")
        if not gps:
            return None

        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode()
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode()
        if not lat or not lon:
            return None

        latitude = _dms_to_decimal(lat, lat_ref)
        longitude = _dms_to_decimal(lon, lon_ref)

        altitude = None
        alt = gps.get(piexif.GPSIFD.GPSAltitude)
        if alt is not None:
            altitude = round(_ratio_to_float(alt), 2)

        return GeoPoint(latitude=latitude, longitude=longitude, altitude=altitude)
    except Exception as exc:  # pragma: no cover - depends on input EXIF
        logger.debug("gps_exif_extraction_failed", error=str(exc))
        return None


def extract_gps_from_pillow(image_bytes: bytes) -> GeoPoint | None:
    """Fallback GPS extraction using Pillow's EXIF reader.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        A :class:`GeoPoint`, or ``None`` if unavailable.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None
        gps_info: dict[str, Any] = {}
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                for key, val in value.items():
                    gps_info[GPSTAGS.get(key, key)] = val
        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None
        latitude = _dms_to_decimal(
            gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N")
        )
        longitude = _dms_to_decimal(
            gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E")
        )
        return GeoPoint(latitude=latitude, longitude=longitude)
    except Exception as exc:  # pragma: no cover
        logger.debug("pillow_gps_extraction_failed", error=str(exc))
        return None


def haversine_meters(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two points in metres.

    Args:
        a: First coordinate.
        b: Second coordinate.

    Returns:
        Distance in metres.
    """
    from math import asin, cos, radians, sin, sqrt

    earth_radius_m = 6_371_000.0
    dlat = radians(b.latitude - a.latitude)
    dlon = radians(b.longitude - a.longitude)
    h = (
        sin(dlat / 2) ** 2
        + cos(radians(a.latitude)) * cos(radians(b.latitude)) * sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_m * asin(sqrt(h))


def build_geojson(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a GeoJSON ``FeatureCollection`` from detection records.

    Args:
        features: Records, each containing ``latitude``, ``longitude`` and
            arbitrary ``properties``.

    Returns:
        A GeoJSON FeatureCollection dictionary.
    """
    collection: list[dict[str, Any]] = []
    for f in features:
        if f.get("latitude") is None or f.get("longitude") is None:
            continue
        collection.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [f["longitude"], f["latitude"]],
                },
                "properties": f.get("properties", {}),
            }
        )
    return {"type": "FeatureCollection", "features": collection}
