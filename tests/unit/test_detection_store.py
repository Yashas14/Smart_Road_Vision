"""Unit tests for the offline SQLite detection store."""

from __future__ import annotations

import pytest

from src.detection.types import FrameResult, SeverityLevel, UrgencyTag
from src.storage.detection_store import DetectionStore
from tests.conftest import make_detection


@pytest.fixture
def store() -> DetectionStore:
    """A fresh in-memory store per test."""
    return DetectionStore(":memory:")


def _frame(score: float = 80.0, *, sev: SeverityLevel = SeverityLevel.HIGH) -> FrameResult:
    det = make_detection(box=(10, 10, 110, 110))
    det.severity_level = sev
    det.severity_score = 0.7
    det.urgency = UrgencyTag.URGENT
    return FrameResult(
        detections=[det],
        image_width=640,
        image_height=480,
        processing_time_ms=10.0,
        model_version="opencv-heuristic-v1",
        road_condition_score=score,
    )


def test_save_and_count(store: DetectionStore) -> None:
    assert store.count() == 0
    det_id = store.save_detection(_frame(), source="image")
    assert det_id > 0
    assert store.count() == 1


def test_save_persists_anomalies(store: DetectionStore) -> None:
    det_id = store.save_detection(_frame(), source="image")
    record = store.get_detection(det_id)
    assert record is not None
    assert record["anomaly_count"] == 1
    assert len(record["anomalies"]) == 1
    assert record["anomalies"][0]["class_name"] == "pothole"


def test_list_detections_orders_recent_first(store: DetectionStore) -> None:
    store.save_detection(_frame(score=90.0), source="image")
    store.save_detection(_frame(score=40.0), source="batch")
    rows = store.list_detections()
    assert len(rows) == 2
    assert rows[0]["source"] == "batch"


def test_list_detections_filters_by_source(store: DetectionStore) -> None:
    store.save_detection(_frame(), source="image")
    store.save_detection(_frame(), source="video")
    assert len(store.list_detections(source="video")) == 1


def test_summary_aggregates(store: DetectionStore) -> None:
    store.save_detection(
        _frame(score=60.0, sev=SeverityLevel.CRITICAL),
        source="image",
        estimated_repair_cost=120.0,
        currency="USD",
    )
    store.save_detection(_frame(score=80.0), source="image", estimated_repair_cost=50.0)
    summary = store.summary()
    assert summary["total_detections"] == 2
    assert summary["total_anomalies"] == 2
    assert summary["avg_road_score"] == 70.0
    assert summary["total_repair_cost"] == 170.0
    assert summary["critical_count"] == 1
    assert summary["by_class"]["pothole"] == 2


def test_geojson_only_geotagged(store: DetectionStore) -> None:
    store.save_detection(_frame(), source="image")  # no GPS
    store.save_detection(_frame(), source="image", latitude=12.97, longitude=77.59)
    gj = store.geojson()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    assert gj["features"][0]["geometry"]["coordinates"] == [77.59, 12.97]


def test_timeline_chronological(store: DetectionStore) -> None:
    store.save_detection(_frame(score=70.0), source="image")
    store.save_detection(_frame(score=50.0), source="image")
    timeline = store.timeline()
    assert [t["road_condition_score"] for t in timeline] == [70.0, 50.0]


def test_clear_removes_all(store: DetectionStore) -> None:
    store.save_detection(_frame(), source="image")
    removed = store.clear()
    assert removed == 1
    assert store.count() == 0
    assert store.all_anomaly_records() == []
