"""SQLModel ORM models and async engine/session helpers.

Defines the persisted entities (Detection, Anomaly, Location, Report) and an
async SQLAlchemy engine configured for PostgreSQL + PostGIS. Geospatial columns
are stored as PostGIS ``POINT`` geometries via GeoAlchemy2.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import Column, Index
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import Field, Relationship, SQLModel

from src.core.config import get_settings

settings = get_settings()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Location(SQLModel, table=True):
    """A geographic location associated with one or more detections."""

    __tablename__ = "locations"

    id: int | None = Field(default=None, primary_key=True)
    latitude: float = Field(index=True)
    longitude: float = Field(index=True)
    altitude: float | None = None
    road_name: str | None = Field(default=None, index=True)
    city: str | None = None
    geom: Any | None = Field(
        default=None,
        sa_column=Column(Geometry(geometry_type="POINT", srid=4326)),
    )
    created_at: datetime = Field(default_factory=_utcnow)

    detections: list["Detection"] = Relationship(back_populates="location")


class Detection(SQLModel, table=True):
    """A single processed image/frame and its aggregate metadata."""

    __tablename__ = "detections"

    id: int | None = Field(default=None, primary_key=True)
    source: str = Field(default="image", index=True)  # image | video | stream
    model_version: str = Field(default="yolov11-pothole-v2.0")
    image_width: int = 0
    image_height: int = 0
    anomaly_count: int = Field(default=0, index=True)
    road_condition_score: float = 100.0
    processing_time_ms: float = 0.0
    location_id: int | None = Field(default=None, foreign_key="locations.id")
    created_at: datetime = Field(default_factory=_utcnow, index=True)

    location: Location | None = Relationship(back_populates="detections")
    anomalies: list["Anomaly"] = Relationship(
        back_populates="detection",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Anomaly(SQLModel, table=True):
    """An individual anomaly belonging to a detection."""

    __tablename__ = "anomalies"

    id: int | None = Field(default=None, primary_key=True)
    detection_id: int = Field(foreign_key="detections.id", index=True)
    class_name: str = Field(index=True)
    confidence: float = 0.0
    severity_score: float = Field(default=0.0, index=True)
    severity_level: str = Field(default="LOW", index=True)
    urgency: str = Field(default="MONITOR")
    depth_mm: float | None = None
    area_px: float = 0.0
    area_m2: float | None = None
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    track_id: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    detection: Detection | None = Relationship(back_populates="anomalies")


class Report(SQLModel, table=True):
    """A generated maintenance report referencing a set of detections."""

    __tablename__ = "reports"

    id: int | None = Field(default=None, primary_key=True)
    title: str = "Road Condition Report"
    file_path: str | None = None
    total_anomalies: int = 0
    avg_road_score: float = 100.0
    date_from: datetime | None = None
    date_to: datetime | None = None
    status: str = Field(default="completed", index=True)
    created_at: datetime = Field(default_factory=_utcnow, index=True)


# Composite/geo indexes (created in addition to the SQL migration).
Index("ix_anomalies_severity_class", Anomaly.severity_level, Anomaly.class_name)


# --------------------------------------------------------------------------- #
# Async engine / session management
# --------------------------------------------------------------------------- #

_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

_session_factory = async_sessionmaker(
    bind=_engine, class_=AsyncSession, expire_on_commit=False
)


def get_engine():
    """Return the shared async SQLAlchemy engine."""
    return _engine


async def init_db() -> None:
    """Create tables and enable the PostGIS extension if needed."""
    from sqlalchemy import text

    async with _engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (FastAPI dependency)."""
    async with _session_factory() as session:
        yield session
