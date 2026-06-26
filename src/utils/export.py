"""Export detections to tabular and geospatial formats.

Provides lightweight, dependency-friendly serialisation of detection results to
CSV, GeoJSON and pandas DataFrames so results can be handed to GIS tools,
spreadsheets or downstream analytics without the database layer.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from typing import Any

from src.detection.types import AnomalyDetection, FrameResult

CSV_COLUMNS: list[str] = [
    "frame_index",
    "class_name",
    "confidence",
    "severity_level",
    "severity_score",
    "urgency",
    "depth_mm",
    "area_px",
    "x1",
    "y1",
    "x2",
    "y2",
    "latitude",
    "longitude",
]


def detection_to_record(
    det: AnomalyDetection,
    frame_index: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Flatten a single :class:`AnomalyDetection` into a flat record.

    Args:
        det: Detection to flatten.
        frame_index: Optional source frame index.
        latitude: Optional geotag latitude.
        longitude: Optional geotag longitude.

    Returns:
        A flat ``dict`` suitable for CSV/DataFrame rows.
    """
    x1, y1, x2, y2 = det.bbox.xyxy()
    return {
        "frame_index": frame_index if frame_index is not None else det.track_id,
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 4),
        "severity_level": str(det.severity_level),
        "severity_score": round(float(det.severity_score), 4),
        "urgency": str(det.urgency),
        "depth_mm": det.depth_mm,
        "area_px": round(float(det.area_px), 2),
        "x1": round(float(x1), 1),
        "y1": round(float(y1), 1),
        "x2": round(float(x2), 1),
        "y2": round(float(y2), 1),
        "latitude": latitude,
        "longitude": longitude,
    }


def frame_results_to_records(frames: Iterable[FrameResult]) -> list[dict[str, Any]]:
    """Flatten every detection across multiple frames into a list of records.

    Args:
        frames: Iterable of :class:`FrameResult`.

    Returns:
        A list of flat detection records.
    """
    records: list[dict[str, Any]] = []
    for frame in frames:
        for det in frame.detections:
            records.append(detection_to_record(det, frame_index=frame.frame_index))
    return records


def detections_to_csv(detections: Sequence[AnomalyDetection]) -> str:
    """Serialise detections to a CSV string.

    Args:
        detections: Detections to serialise.

    Returns:
        CSV text with a header row.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for det in detections:
        writer.writerow(detection_to_record(det))
    return buffer.getvalue()


def records_to_geojson(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a GeoJSON ``FeatureCollection`` from geotagged records.

    Records missing ``latitude``/``longitude`` are skipped.

    Args:
        records: Flat detection records (e.g. from :func:`detection_to_record`).

    Returns:
        A GeoJSON ``FeatureCollection`` dictionary.
    """
    features: list[dict[str, Any]] = []
    for rec in records:
        lat, lon = rec.get("latitude"), rec.get("longitude")
        if lat is None or lon is None:
            continue
        properties = {k: v for k, v in rec.items() if k not in {"latitude", "longitude"}}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}
