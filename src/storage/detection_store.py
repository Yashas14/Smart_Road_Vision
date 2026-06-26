"""SQLite-backed detection store for offline persistence and analytics.

Stores one row per analysed image/frame in ``detections`` plus one row per
anomaly in ``anomalies``. Provides aggregation helpers (summary, GeoJSON,
timeline) that power the analytics dashboard, map view and reports without
needing PostgreSQL/PostGIS.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT
from src.core.logging import get_logger
from src.detection.types import FrameResult

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    anomaly_count INTEGER NOT NULL,
    road_condition_score REAL,
    processing_time_ms REAL,
    model_version TEXT,
    estimated_repair_cost REAL,
    currency TEXT,
    latitude REAL,
    longitude REAL,
    thumbnail TEXT
);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL,
    severity_level TEXT,
    severity_score REAL,
    urgency TEXT,
    depth_mm REAL,
    area_px REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_anom_det ON anomalies(detection_id);
CREATE INDEX IF NOT EXISTS idx_det_created ON detections(created_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DetectionStore:
    """Thread-safe SQLite store for detections and anomalies.

    Args:
        db_path: Filesystem path to the SQLite database, or ``":memory:"``.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- writes -------------------------------------------------------------
    def save_detection(
        self,
        frame_result: FrameResult,
        source: str = "image",
        latitude: float | None = None,
        longitude: float | None = None,
        estimated_repair_cost: float | None = None,
        currency: str | None = None,
        thumbnail: str | None = None,
    ) -> int:
        """Persist a frame result and its anomalies; returns the detection id."""
        created = _now()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO detections
                   (created_at, source, image_width, image_height, anomaly_count,
                    road_condition_score, processing_time_ms, model_version,
                    estimated_repair_cost, currency, latitude, longitude, thumbnail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    created,
                    source,
                    frame_result.image_width,
                    frame_result.image_height,
                    frame_result.count,
                    frame_result.road_condition_score,
                    frame_result.processing_time_ms,
                    frame_result.model_version,
                    estimated_repair_cost,
                    currency,
                    latitude,
                    longitude,
                    thumbnail,
                ),
            )
            det_id = int(cur.lastrowid or 0)
            for a in frame_result.detections:
                x1, y1, x2, y2 = a.bbox.xyxy()
                self._conn.execute(
                    """INSERT INTO anomalies
                       (detection_id, created_at, class_name, confidence,
                        severity_level, severity_score, urgency, depth_mm, area_px,
                        x1, y1, x2, y2)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        det_id,
                        created,
                        a.class_name,
                        a.confidence,
                        str(a.severity_level),
                        a.severity_score,
                        str(a.urgency),
                        a.depth_mm,
                        a.area_px,
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )
            self._conn.commit()
        return det_id

    def clear(self) -> int:
        """Delete all records; returns the number of detections removed."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            self._conn.execute("DELETE FROM anomalies")
            self._conn.execute("DELETE FROM detections")
            self._conn.commit()
        return int(n)

    # -- reads --------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0])

    def list_detections(
        self,
        limit: int = 50,
        offset: int = 0,
        severity: str | None = None,
        source: str | None = None,
        include_thumbnail: bool = False,
    ) -> list[dict[str, Any]]:
        """Return detection rows, most recent first."""
        cols = (
            "id, created_at, source, image_width, image_height, anomaly_count, "
            "road_condition_score, processing_time_ms, model_version, "
            "estimated_repair_cost, currency, latitude, longitude"
        )
        if include_thumbnail:
            cols += ", thumbnail"
        sql = f"SELECT {cols} FROM detections"
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if severity:
            clauses.append("id IN (SELECT detection_id FROM anomalies WHERE severity_level = ?)")
            params.append(severity.upper())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_detection(self, det_id: int) -> dict[str, Any] | None:
        """Return a single detection with its anomalies."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
            if row is None:
                return None
            anoms = self._conn.execute(
                "SELECT * FROM anomalies WHERE detection_id = ?", (det_id,)
            ).fetchall()
        data = dict(row)
        data["anomalies"] = [dict(a) for a in anoms]
        return data

    def all_anomaly_records(self) -> list[dict[str, Any]]:
        """Return all anomalies as flat records (for statistics/reports)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM anomalies").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        """Compute aggregate analytics across all stored detections."""
        with self._lock:
            det_rows = self._conn.execute(
                "SELECT road_condition_score, estimated_repair_cost, currency, "
                "latitude, longitude, source, created_at FROM detections"
            ).fetchall()
            anom_rows = self._conn.execute(
                "SELECT class_name, severity_level, urgency, confidence, "
                "severity_score, depth_mm FROM anomalies"
            ).fetchall()

        total_detections = len(det_rows)
        total_anomalies = len(anom_rows)
        road_scores = [
            r["road_condition_score"] for r in det_rows if r["road_condition_score"] is not None
        ]
        costs = [
            r["estimated_repair_cost"] for r in det_rows if r["estimated_repair_cost"] is not None
        ]
        confs = [r["confidence"] for r in anom_rows if r["confidence"] is not None]
        sev_scores = [r["severity_score"] for r in anom_rows if r["severity_score"] is not None]

        by_class = Counter(r["class_name"] for r in anom_rows)
        by_severity = Counter(r["severity_level"] for r in anom_rows)
        by_urgency = Counter(r["urgency"] for r in anom_rows)
        currency = next((r["currency"] for r in det_rows if r["currency"]), "USD")

        return {
            "total_detections": total_detections,
            "total_anomalies": total_anomalies,
            "avg_road_score": round(sum(road_scores) / len(road_scores), 1)
            if road_scores
            else 100.0,
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "avg_severity_score": round(sum(sev_scores) / len(sev_scores), 3)
            if sev_scores
            else 0.0,
            "total_repair_cost": round(sum(costs), 2),
            "currency": currency,
            "critical_count": int(by_severity.get("CRITICAL", 0)),
            "geotagged_count": sum(1 for r in det_rows if r["latitude"] is not None),
            "by_class": dict(by_class),
            "by_severity": dict(by_severity),
            "by_urgency": dict(by_urgency),
        }

    def geojson(self) -> dict[str, Any]:
        """Return geotagged detections as a GeoJSON FeatureCollection."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, latitude, longitude, anomaly_count, road_condition_score, "
                "estimated_repair_cost, currency, created_at, source "
                "FROM detections WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
            ).fetchall()
        features = []
        for r in rows:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [r["longitude"], r["latitude"]],
                    },
                    "properties": {
                        "id": r["id"],
                        "anomaly_count": r["anomaly_count"],
                        "road_condition_score": r["road_condition_score"],
                        "estimated_repair_cost": r["estimated_repair_cost"],
                        "currency": r["currency"],
                        "created_at": r["created_at"],
                        "source": r["source"],
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def timeline(self) -> list[dict[str, Any]]:
        """Return per-detection time-series points (chronological)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT created_at, anomaly_count, road_condition_score "
                "FROM detections ORDER BY datetime(created_at) ASC, id ASC"
            ).fetchall()
        return [
            {
                "created_at": r["created_at"],
                "anomaly_count": r["anomaly_count"],
                "road_condition_score": r["road_condition_score"],
            }
            for r in rows
        ]


_store: DetectionStore | None = None
_store_lock = threading.Lock()


def get_store() -> DetectionStore:
    """Return the process-wide detection store singleton.

    Uses a file-backed SQLite database under ``data/`` so history survives
    restarts. Falls back to an in-memory database if the path is unwritable.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                try:
                    data_dir = PROJECT_ROOT / "data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    _store = DetectionStore(data_dir / "smartroadvision.db")
                    logger.info("detection_store_ready", path=str(_store.db_path))
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    logger.warning("detection_store_memory_fallback", error=str(exc))
                    _store = DetectionStore(":memory:")
    return _store
