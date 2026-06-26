"""Integration tests for analytics endpoints and batch detection.

Uses an in-memory store injected via monkeypatch so tests stay isolated from the
on-disk database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from tests.conftest import make_detection

from src.api.dependencies import get_db, get_image_pipeline
from src.api.main import app
from src.detection.postprocessor import draw_annotations
from src.detection.types import FrameResult, SeverityLevel, UrgencyTag
from src.storage.detection_store import DetectionStore


class _FakePipeline:
    def __init__(self) -> None:
        self.detector = type("D", (), {"is_loaded": True})()

    def process(self, image: np.ndarray, denoise: bool = False, annotate: bool = True):
        h, w = image.shape[:2]
        det = make_detection(box=(10, 10, 110, 110))
        det.severity_level = SeverityLevel.HIGH
        det.severity_score = 0.6
        det.urgency = UrgencyTag.URGENT
        result = FrameResult(
            detections=[det],
            image_width=w,
            image_height=h,
            processing_time_ms=7.0,
            model_version="opencv-heuristic-v1",
            road_condition_score=75.0,
        )
        annotated = draw_annotations(image, result.detections) if annotate else None
        return result, annotated


async def _fake_db() -> AsyncIterator[None]:
    yield None


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> DetectionStore:
    s = DetectionStore(":memory:")
    monkeypatch.setattr("src.api.routers.analytics.get_store", lambda: s)
    monkeypatch.setattr("src.api.routers.detection.get_store", lambda: s)
    return s


@pytest.fixture
def client(store: DetectionStore) -> AsyncClient:
    app.dependency_overrides[get_image_pipeline] = lambda: _FakePipeline()
    app.dependency_overrides[get_db] = _fake_db
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _encode_image() -> bytes:
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.mark.asyncio
async def test_empty_summary(client: AsyncClient) -> None:
    async with client:
        resp = await client.get("/api/v1/analytics/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_detections"] == 0
    assert body["avg_road_score"] == 100.0


@pytest.mark.asyncio
async def test_detect_persists_to_store(client: AsyncClient, store: DetectionStore) -> None:
    files = {"file": ("road.jpg", _encode_image(), "image/jpeg")}
    data = {"persist": "true", "latitude": "12.97", "longitude": "77.59"}
    async with client:
        resp = await client.post("/api/v1/detect/image", files=files, data=data)
        assert resp.status_code == 200
        summary = (await client.get("/api/v1/analytics/summary")).json()
        geojson = (await client.get("/api/v1/analytics/geojson")).json()
    assert store.count() == 1
    assert summary["total_detections"] == 1
    assert summary["total_anomalies"] == 1
    assert len(geojson["features"]) == 1


@pytest.mark.asyncio
async def test_batch_detection(client: AsyncClient, store: DetectionStore) -> None:
    img = _encode_image()
    files = [
        ("files", ("a.jpg", img, "image/jpeg")),
        ("files", ("b.jpg", img, "image/jpeg")),
    ]
    async with client:
        resp = await client.post("/api/v1/detect/batch", files=files, data={"annotate": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_images"] == 2
    assert body["succeeded"] == 2
    assert body["total_anomalies"] == 2
    assert store.count() == 2


@pytest.mark.asyncio
async def test_history_and_clear(client: AsyncClient, store: DetectionStore) -> None:
    files = {"file": ("road.jpg", _encode_image(), "image/jpeg")}
    async with client:
        await client.post("/api/v1/detect/image", files=files, data={"persist": "true"})
        history = (await client.get("/api/v1/analytics/history")).json()
        assert len(history) == 1
        cleared = (await client.delete("/api/v1/analytics/history")).json()
        assert cleared["removed"] == 1
        after = (await client.get("/api/v1/analytics/history")).json()
    assert after == []
