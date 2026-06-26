"""Integration tests for PDF report generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import get_settings
from src.reporting.report_generator import ReportGenerator
from src.reporting.statistics import DetectionStatistics


@pytest.fixture
def report_generator(tmp_path: Path) -> ReportGenerator:
    settings = get_settings()
    settings.reports_dir = tmp_path
    return ReportGenerator(settings)


def _sample_records() -> list[dict]:
    return [
        {
            "class_name": "pothole",
            "confidence": 0.91,
            "severity_score": 0.85,
            "severity_level": "CRITICAL",
            "urgency": "IMMEDIATE",
            "depth_mm": 120.0,
        },
        {
            "class_name": "crack",
            "confidence": 0.7,
            "severity_score": 0.4,
            "severity_level": "MEDIUM",
            "urgency": "SCHEDULE_REPAIR",
            "depth_mm": 20.0,
        },
    ]


def _sample_gallery() -> list[bytes]:
    img = np.full((120, 160, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return [buf.tobytes()]


def test_statistics_summary() -> None:
    stats = DetectionStatistics(_sample_records())
    summary = stats.summary(road_scores=[80.0, 60.0])
    assert summary.total_anomalies == 2
    assert summary.by_severity["CRITICAL"] == 1
    assert summary.by_class["pothole"] == 1
    assert summary.avg_road_score == 70.0
    # Repair queue ranks IMMEDIATE first.
    assert summary.repair_priority_queue[0]["urgency"] == "IMMEDIATE"


def test_statistics_empty() -> None:
    stats = DetectionStatistics([])
    summary = stats.summary()
    assert summary.total_anomalies == 0
    assert summary.avg_road_score == 100.0


def test_report_generation_creates_pdf(report_generator: ReportGenerator) -> None:
    path = report_generator.generate(
        title="Test Road Report",
        anomaly_records=_sample_records(),
        road_scores=[80.0, 60.0],
        gallery_images=_sample_gallery(),
        geo_summary={"center": "12.97, 77.59", "radius_m": 500},
    )
    assert path.exists()
    assert path.suffix == ".pdf"
    assert path.stat().st_size > 1000  # non-trivial PDF
    header = path.read_bytes()[:5]
    assert header.startswith(b"%PDF")


def test_report_generation_empty_records(report_generator: ReportGenerator) -> None:
    path = report_generator.generate(
        title="Empty Report",
        anomaly_records=[],
        road_scores=[],
    )
    assert path.exists()
    assert path.read_bytes()[:5].startswith(b"%PDF")
