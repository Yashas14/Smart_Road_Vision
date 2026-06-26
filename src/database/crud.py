"""Async CRUD operations including PostGIS geospatial queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.database.models import Anomaly, Detection, Location, Report
from src.detection.types import FrameResult
from src.utils.geo_utils import GeoPoint

logger = get_logger(__name__)


async def create_location(
    session: AsyncSession, point: GeoPoint, road_name: str | None = None
) -> Location:
    """Persist a geographic location with a PostGIS POINT geometry.

    Args:
        session: Active async session.
        point: Coordinate to store.
        road_name: Optional road name.

    Returns:
        The persisted :class:`Location`.
    """
    location = Location(
        latitude=point.latitude,
        longitude=point.longitude,
        altitude=point.altitude,
        road_name=road_name,
        geom=func.ST_SetSRID(
            func.ST_MakePoint(point.longitude, point.latitude), 4326
        ),
    )
    session.add(location)
    await session.flush()
    return location


async def save_frame_result(
    session: AsyncSession,
    result: FrameResult,
    source: str = "image",
    point: GeoPoint | None = None,
) -> Detection:
    """Persist a frame result and all of its anomalies.

    Args:
        session: Active async session.
        result: The frame result to persist.
        source: Origin of the detection (``image``/``video``/``stream``).
        point: Optional GPS coordinate to associate.

    Returns:
        The persisted :class:`Detection` with anomalies attached.
    """
    location_id = None
    if point is not None:
        location = await create_location(session, point)
        location_id = location.id

    detection = Detection(
        source=source,
        model_version=result.model_version,
        image_width=result.image_width,
        image_height=result.image_height,
        anomaly_count=result.count,
        road_condition_score=result.road_condition_score or 100.0,
        processing_time_ms=result.processing_time_ms,
        location_id=location_id,
    )
    session.add(detection)
    await session.flush()

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox.xyxy()
        session.add(
            Anomaly(
                detection_id=detection.id,
                class_name=det.class_name,
                confidence=det.confidence,
                severity_score=det.severity_score,
                severity_level=str(det.severity_level),
                urgency=str(det.urgency),
                depth_mm=det.depth_mm,
                area_px=det.area_px,
                area_m2=det.area_m2,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                track_id=det.track_id,
            )
        )

    await session.commit()
    await session.refresh(detection)
    logger.info("detection_saved", detection_id=detection.id, count=result.count)
    return detection


async def list_detections(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    severity: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[Detection]:
    """List detections with pagination and optional filters.

    Args:
        session: Active async session.
        limit: Maximum rows to return.
        offset: Pagination offset.
        severity: Filter by anomaly severity level.
        date_from: Inclusive lower bound on ``created_at``.
        date_to: Inclusive upper bound on ``created_at``.

    Returns:
        Matching detections, newest first, with anomalies eagerly loaded.
    """
    stmt = (
        select(Detection)
        .options(selectinload(Detection.anomalies), selectinload(Detection.location))
        .order_by(Detection.created_at.desc())
    )
    if date_from is not None:
        stmt = stmt.where(Detection.created_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Detection.created_at <= date_to)
    if severity is not None:
        stmt = stmt.where(
            Detection.id.in_(
                select(Anomaly.detection_id).where(
                    Anomaly.severity_level == severity.upper()
                )
            )
        )
    stmt = stmt.limit(limit).offset(offset)
    rows = await session.execute(stmt)
    return list(rows.scalars().unique().all())


async def get_detection(session: AsyncSession, detection_id: int) -> Detection | None:
    """Fetch a single detection by id with its anomalies and location."""
    stmt = (
        select(Detection)
        .where(Detection.id == detection_id)
        .options(selectinload(Detection.anomalies), selectinload(Detection.location))
    )
    row = await session.execute(stmt)
    return row.scalars().first()


async def find_critical_within_radius(
    session: AsyncSession,
    latitude: float,
    longitude: float,
    radius_m: float = 500.0,
) -> list[dict[str, Any]]:
    """Find CRITICAL anomalies within a radius of a point (PostGIS query).

    Args:
        session: Active async session.
        latitude: Centre latitude.
        longitude: Centre longitude.
        radius_m: Search radius in metres.

    Returns:
        A list of dicts describing nearby critical anomalies.
    """
    query = text(
        """
        SELECT a.id, a.class_name, a.severity_level, a.confidence,
               l.latitude, l.longitude,
               ST_Distance(
                   l.geom::geography,
                   ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
               ) AS distance_m
        FROM anomalies a
        JOIN detections d ON a.detection_id = d.id
        JOIN locations l ON d.location_id = l.id
        WHERE a.severity_level = 'CRITICAL'
          AND ST_DWithin(
                  l.geom::geography,
                  ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                  :radius
              )
        ORDER BY distance_m ASC
        """
    )
    rows = await session.execute(
        query, {"lat": latitude, "lon": longitude, "radius": radius_m}
    )
    return [dict(r._mapping) for r in rows]


async def severity_breakdown(session: AsyncSession) -> dict[str, int]:
    """Return anomaly counts grouped by severity level."""
    stmt = select(Anomaly.severity_level, func.count()).group_by(
        Anomaly.severity_level
    )
    rows = await session.execute(stmt)
    return {level: count for level, count in rows.all()}


async def create_report(session: AsyncSession, report: Report) -> Report:
    """Persist a report record."""
    session.add(report)
    await session.commit()
    await session.refresh(report)
    return report


async def get_report(session: AsyncSession, report_id: int) -> Report | None:
    """Fetch a report by id."""
    row = await session.execute(select(Report).where(Report.id == report_id))
    return row.scalars().first()
