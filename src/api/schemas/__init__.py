"""Pydantic v2 request/response schemas for the API."""

from src.api.schemas.detection import (
    AnomalyResult,
    BBox,
    DetectionResponse,
    GeoCoordinate,
    HealthResponse,
)
from src.api.schemas.report import ReportRequest, ReportResponse

__all__ = [
    "BBox",
    "GeoCoordinate",
    "AnomalyResult",
    "DetectionResponse",
    "HealthResponse",
    "ReportRequest",
    "ReportResponse",
]
