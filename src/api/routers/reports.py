"""Report generation and download endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_app_settings, get_db
from src.api.schemas.report import ReportRequest, ReportResponse
from src.core.config import Settings
from src.core.exceptions import SmartRoadVisionError
from src.core.logging import get_logger
from src.database import crud
from src.database.models import Report
from src.reporting.report_generator import ReportGenerator
from src.storage import get_store

logger = get_logger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])


def _records_from_detections(detections: list[Any]) -> list[dict[str, Any]]:
    """Flatten detection ORM rows into anomaly records for statistics."""
    records: list[dict[str, Any]] = []
    for d in detections:
        for a in d.anomalies:
            records.append(
                {
                    "class_name": a.class_name,
                    "confidence": a.confidence,
                    "severity_score": a.severity_score,
                    "severity_level": a.severity_level,
                    "urgency": a.urgency,
                    "depth_mm": a.depth_mm,
                    "created_at": a.created_at,
                }
            )
    return records


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ReportResponse:
    """Generate a PDF maintenance report over a filtered set of detections."""
    detections = await crud.list_detections(
        session,
        limit=1000,
        severity=request.severity,
        date_from=request.date_from,
        date_to=request.date_to,
    )
    records = _records_from_detections(detections)
    road_scores = [d.road_condition_score for d in detections]

    generator = ReportGenerator(settings)
    try:
        pdf_path = generator.generate(
            title=request.title,
            anomaly_records=records,
            road_scores=road_scores,
            gallery_images=None,
        )
    except SmartRoadVisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    avg_score = round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0
    report = await crud.create_report(
        session,
        Report(
            title=request.title,
            file_path=str(pdf_path),
            total_anomalies=len(records),
            avg_road_score=avg_score,
            date_from=request.date_from,
            date_to=request.date_to,
            status="completed",
        ),
    )

    return ReportResponse(
        report_id=report.id,
        title=report.title,
        status=report.status,
        total_anomalies=report.total_anomalies,
        avg_road_score=report.avg_road_score,
        file_path=report.file_path,
        download_url=f"{settings.api_v1_prefix}/reports/{report.id}/download",
        created_at=report.created_at,
    )


@router.post("/offline/generate")
async def generate_report_offline(
    settings: Annotated[Settings, Depends(get_app_settings)],
    title: str = "Road Maintenance Report",
) -> FileResponse:
    """Generate a PDF report from the offline store and return it directly.

    This path requires no database and works fully offline using the SQLite
    detection store.
    """
    store = get_store()
    records = store.all_anomaly_records()
    detections = store.list_detections(limit=1000)
    road_scores = [
        d["road_condition_score"] for d in detections if d.get("road_condition_score") is not None
    ]
    if not records:
        raise HTTPException(
            status_code=404,
            detail="No detections stored yet. Analyse images first.",
        )

    generator = ReportGenerator(settings)
    try:
        pdf_path = generator.generate(
            title=title,
            anomaly_records=records,
            road_scores=road_scores,
            gallery_images=None,
        )
    except SmartRoadVisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    path = Path(pdf_path)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: int,
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ReportResponse:
    """Fetch metadata for a previously generated report."""
    report = await crud.get_report(session, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportResponse(
        report_id=report.id,
        title=report.title,
        status=report.status,
        total_anomalies=report.total_anomalies,
        avg_road_score=report.avg_road_score,
        file_path=report.file_path,
        download_url=f"{settings.api_v1_prefix}/reports/{report.id}/download",
        created_at=report.created_at,
    )


@router.get("/{report_id}/download")
async def download_report(
    report_id: int,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    """Stream the generated PDF report file."""
    report = await crud.get_report(session, report_id)
    if report is None or not report.file_path:
        raise HTTPException(status_code=404, detail="Report not found")
    path = report.file_path
    if not await anyio.Path(path).is_file():
        raise HTTPException(status_code=410, detail="Report file no longer available")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
