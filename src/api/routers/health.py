"""Health and Prometheus metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.api.dependencies import get_detector_engine_info, get_detector_status
from src.api.schemas.detection import HealthResponse
from src.core.config import get_settings

router = APIRouter(tags=["health"])

# Prometheus metrics (imported and incremented elsewhere as needed).
REQUEST_COUNT = Counter(
    "smartroad_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
DETECTION_LATENCY = Histogram(
    "smartroad_detection_latency_seconds",
    "Detection processing latency in seconds",
)
ANOMALY_COUNT = Counter(
    "smartroad_anomalies_total",
    "Total anomalies detected",
    ["class_name", "severity"],
)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe."""
    settings = get_settings()
    using_fallback, engine = get_detector_engine_info()
    return HealthResponse(
        status="ok",
        version="2.0.0",
        model_loaded=get_detector_status(),
        environment=settings.app_env,
        using_fallback=using_fallback,
        engine=engine,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics in text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
