"""Integration tests for the FastAPI detection endpoints.

Heavy models and the database are replaced with fakes via dependency overrides
so these tests run fully offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_db, get_image_pipeline
from src.api.main import app
from src.detection.postprocessor import draw_annotations
from src.detection.types import FrameResult
from tests.conftest import make_detection


class _FakePipeline:
    """Fake image pipeline returning a single fixed detection."""

    def __init__(self) -> None:
        self.detector = type("D", (), {"is_loaded": True})()

    def process(self, image: np.ndarray, denoise: bool = False, annotate: bool = True):
        h, w = image.shape[:2]
        det = make_detection(box=(10, 10, 110, 110))
        result = FrameResult(
            detections=[det],
            image_width=w,
            image_height=h,
            processing_time_ms=7.0,
            model_version="yolov11-test",
            road_condition_score=85.0,
        )
        annotated = draw_annotations(image, result.detections) if annotate else None
        return result, annotated


async def _fake_db() -> AsyncIterator[None]:
    yield None


@pytest.fixture
def client() -> AsyncClient:
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
async def test_health_endpoint(client: AsyncClient) -> None:
    async with client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    async with client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "SmartRoadVision"


@pytest.mark.asyncio
async def test_detect_image_returns_detections(client: AsyncClient) -> None:
    files = {"file": ("road.jpg", _encode_image(), "image/jpeg")}
    data = {"persist": "false", "annotate": "true"}
    async with client:
        resp = await client.post("/api/v1/detect/image", files=files, data=data)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["detections"][0]["class_name"] == "pothole"
    assert body["annotated_image_base64"] is not None
    assert body["road_condition_score"] == 85.0


@pytest.mark.asyncio
async def test_detect_image_rejects_empty_upload(client: AsyncClient) -> None:
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    data = {"persist": "false"}
    async with client:
        resp = await client.post("/api/v1/detect/image", files=files, data=data)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient) -> None:
    async with client:
        resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "smartroad_requests_total" in resp.text


def teardown_module() -> None:
    """Clear dependency overrides after the module's tests run."""
    app.dependency_overrides.clear()
