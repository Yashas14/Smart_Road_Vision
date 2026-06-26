"""Analytics, history and geospatial endpoints backed by the offline store."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from src.core.logging import get_logger
from src.storage import get_store

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def analytics_summary() -> dict[str, Any]:
    """Aggregate statistics across all stored detections."""
    return get_store().summary()


@router.get("/geojson")
async def analytics_geojson() -> dict[str, Any]:
    """Geotagged detections as a GeoJSON FeatureCollection."""
    return get_store().geojson()


@router.get("/timeline")
async def analytics_timeline() -> list[dict[str, Any]]:
    """Chronological per-detection time-series points."""
    return get_store().timeline()


@router.get("/history")
async def analytics_history(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    severity: Annotated[str | None, Query()] = None,
    source: Annotated[str | None, Query()] = None,
    include_thumbnail: Annotated[bool, Query()] = False,
) -> list[dict[str, Any]]:
    """Paginated detection history, most recent first."""
    return get_store().list_detections(
        limit=limit,
        offset=offset,
        severity=severity,
        source=source,
        include_thumbnail=include_thumbnail,
    )


@router.get("/history/{detection_id}")
async def analytics_detection(detection_id: int) -> dict[str, Any]:
    """A single stored detection with its anomalies and thumbnail."""
    record = get_store().get_detection(detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return record


@router.delete("/history")
async def clear_history() -> dict[str, int]:
    """Delete all stored detections; returns how many were removed."""
    removed = get_store().clear()
    return {"removed": removed}
