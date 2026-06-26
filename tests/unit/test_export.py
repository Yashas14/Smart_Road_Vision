"""Unit tests for :mod:`src.utils.export`."""

from __future__ import annotations

import csv
import io

from src.detection.types import FrameResult
from src.utils.export import (
    CSV_COLUMNS,
    detection_to_record,
    detections_to_csv,
    frame_results_to_records,
    records_to_geojson,
)
from tests.conftest import make_detection


def test_detection_to_record_has_all_columns() -> None:
    rec = detection_to_record(make_detection(), frame_index=3)
    for column in CSV_COLUMNS:
        assert column in rec
    assert rec["frame_index"] == 3
    assert rec["class_name"] == "pothole"
    assert rec["x1"] == 100.0 and rec["x2"] == 200.0


def test_detection_to_record_carries_geotag() -> None:
    rec = detection_to_record(make_detection(), latitude=12.9, longitude=77.5)
    assert rec["latitude"] == 12.9
    assert rec["longitude"] == 77.5


def test_detections_to_csv_round_trips() -> None:
    dets = [make_detection(), make_detection(class_name="crack", class_id=1)]
    text = detections_to_csv(dets)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 2
    assert rows[0]["class_name"] == "pothole"
    assert rows[1]["class_name"] == "crack"
    assert set(CSV_COLUMNS).issubset(rows[0].keys())


def test_detections_to_csv_empty_has_header_only() -> None:
    text = detections_to_csv([])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows == []
    assert text.strip().split("\r\n")[0].startswith("frame_index")


def test_frame_results_to_records_flattens() -> None:
    frame = FrameResult(
        detections=[make_detection(), make_detection(class_name="hump")],
        image_width=640,
        image_height=480,
        processing_time_ms=5.0,
        model_version="v",
        frame_index=7,
    )
    records = frame_results_to_records([frame])
    assert len(records) == 2
    assert all(r["frame_index"] == 7 for r in records)


def test_records_to_geojson_skips_missing_coords() -> None:
    records = [
        detection_to_record(make_detection(), latitude=1.0, longitude=2.0),
        detection_to_record(make_detection()),  # no coords
    ]
    geojson = records_to_geojson(records)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["geometry"]["coordinates"] == [2.0, 1.0]
    assert "latitude" not in feature["properties"]
    assert feature["properties"]["class_name"] == "pothole"
