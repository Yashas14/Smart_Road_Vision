"""Unit tests for geospatial utilities."""

from __future__ import annotations

from src.utils.geo_utils import (
    GeoPoint,
    _dms_to_decimal,
    build_geojson,
    haversine_meters,
)


def test_dms_to_decimal_north() -> None:
    # 12°58'15" N -> ~12.9708
    dms = ((12, 1), (58, 1), (15, 1))
    assert abs(_dms_to_decimal(dms, "N") - 12.9708) < 0.001


def test_dms_to_decimal_south_is_negative() -> None:
    dms = ((12, 1), (0, 1), (0, 1))
    assert _dms_to_decimal(dms, "S") == -12.0


def test_haversine_zero_distance() -> None:
    p = GeoPoint(12.97, 77.59)
    assert haversine_meters(p, p) == 0.0


def test_haversine_known_distance() -> None:
    # Bangalore -> ~1km north
    a = GeoPoint(12.9716, 77.5946)
    b = GeoPoint(12.9806, 77.5946)
    dist = haversine_meters(a, b)
    assert 900 < dist < 1100


def test_build_geojson_structure() -> None:
    features = [
        {
            "latitude": 12.97,
            "longitude": 77.59,
            "properties": {"severity": "CRITICAL"},
        }
    ]
    geojson = build_geojson(features)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["geometry"]["coordinates"] == [77.59, 12.97]
    assert feature["properties"]["severity"] == "CRITICAL"


def test_build_geojson_skips_missing_coords() -> None:
    features = [{"latitude": None, "longitude": 77.59}]
    geojson = build_geojson(features)
    assert geojson["features"] == []


def test_geopoint_to_dict() -> None:
    p = GeoPoint(1.0, 2.0, altitude=3.0)
    assert p.to_dict() == {"latitude": 1.0, "longitude": 2.0, "altitude": 3.0}
