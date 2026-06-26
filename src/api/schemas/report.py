"""Report request/response schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    """Request to generate a maintenance PDF report."""

    title: str = "Road Condition Report"
    date_from: datetime | None = None
    date_to: datetime | None = None
    severity: str | None = Field(
        default=None, description="Optional severity filter (LOW/MEDIUM/HIGH/CRITICAL)"
    )
    include_gallery: bool = True
    max_gallery_images: int = Field(default=20, ge=0, le=100)


class ReportResponse(BaseModel):
    """Response describing a generated report."""

    report_id: int
    title: str
    status: str
    total_anomalies: int
    avg_road_score: float
    file_path: str | None = None
    download_url: str | None = None
    created_at: datetime
