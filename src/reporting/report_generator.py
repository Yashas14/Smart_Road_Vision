"""PDF maintenance report generation using ReportLab Platypus.

Produces a multi-section PDF: executive summary, detection statistics, severity
breakdown, annotated image gallery, geospatial summary, maintenance
recommendations and technical metadata. An HTML/Jinja2 path is also provided for
WeasyPrint-based rendering when richer styling is desired.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.core.config import Settings, get_settings
from src.core.exceptions import ReportGenerationError
from src.core.logging import get_logger
from src.reporting.statistics import DetectionStatistics, StatisticsSummary

logger = get_logger(__name__)

_SEVERITY_HEX = {
    "LOW": colors.HexColor("#00c800"),
    "MEDIUM": colors.HexColor("#e6b800"),
    "HIGH": colors.HexColor("#ff8c00"),
    "CRITICAL": colors.HexColor("#ff0000"),
}


def _recommendation_for(summary: StatisticsSummary) -> list[str]:
    """Generate maintenance recommendations from a statistics summary."""
    recs: list[str] = []
    critical = summary.by_severity.get("CRITICAL", 0)
    high = summary.by_severity.get("HIGH", 0)
    if critical:
        recs.append(
            f"{critical} CRITICAL anomalies require IMMEDIATE intervention "
            "(road closure or emergency patching within 24-48 hours)."
        )
    if high:
        recs.append(
            f"{high} HIGH-severity anomalies should be scheduled for urgent "
            "repair within one week."
        )
    if summary.avg_road_score < 50:
        recs.append(
            "Overall road condition score is poor (<50). A full resurfacing "
            "assessment is recommended for this corridor."
        )
    elif summary.avg_road_score < 75:
        recs.append(
            "Road condition is fair. Prioritise preventive maintenance to avoid "
            "accelerated degradation."
        )
    if not recs:
        recs.append("Road condition is good. Continue routine monitoring.")
    return recs


class ReportGenerator:
    """Generate PDF road-condition reports.

    Args:
        settings: Application settings; resolved from environment if omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.output_dir = Path(self.settings.reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_gallery = self.settings.max_gallery_images
        self.styles = getSampleStyleSheet()
        self._register_styles()

    def _register_styles(self) -> None:
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Title"],
                fontSize=22,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Subtitle",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.grey,
                alignment=TA_CENTER,
                spaceAfter=18,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#1f3a5f"),
                spaceBefore=14,
                spaceAfter=6,
            )
        )

    def _heading(self, text: str) -> Paragraph:
        return Paragraph(text, self.styles["SectionHeading"])

    def _stats_table(self, summary: StatisticsSummary) -> Table:
        data = [
            ["Metric", "Value"],
            ["Total anomalies", str(summary.total_anomalies)],
            ["Average confidence", f"{summary.avg_confidence:.2f}"],
            ["Average severity score", f"{summary.avg_severity_score:.2f}"],
            ["Average road condition score", f"{summary.avg_road_score:.0f} / 100"],
        ]
        table = Table(data, colWidths=[8 * cm, 6 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table

    def _severity_table(self, summary: StatisticsSummary) -> Table:
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        data = [["Severity", "Count"]]
        styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]
        for i, level in enumerate(order, start=1):
            data.append([level, str(summary.by_severity.get(level, 0))])
            styles.append(("TEXTCOLOR", (0, i), (0, i), _SEVERITY_HEX[level]))
        table = Table(data, colWidths=[8 * cm, 6 * cm])
        table.setStyle(TableStyle(styles))
        return table

    def generate(
        self,
        *,
        title: str,
        anomaly_records: list[dict[str, Any]],
        road_scores: list[float] | None = None,
        gallery_images: list[bytes] | None = None,
        geo_summary: dict[str, Any] | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Generate a PDF report and return its path.

        Args:
            title: Report title.
            anomaly_records: Flat anomaly records for statistics.
            road_scores: Optional per-frame road condition scores.
            gallery_images: Optional annotated image bytes (JPEG/PNG).
            geo_summary: Optional geospatial summary block.
            output_name: Optional output filename (without directory).

        Returns:
            Path to the generated PDF file.

        Raises:
            ReportGenerationError: If PDF generation fails.
        """
        stats = DetectionStatistics(anomaly_records)
        summary = stats.summary(road_scores)

        name = output_name or f"road_report_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.pdf"
        out_path = self.output_dir / name

        try:
            doc = SimpleDocTemplate(
                str(out_path),
                pagesize=A4,
                topMargin=2 * cm,
                bottomMargin=2 * cm,
                title=title,
            )
            story: list[Any] = []

            # --- Title ---
            story.append(Paragraph(title, self.styles["ReportTitle"]))
            story.append(
                Paragraph(
                    f"SmartRoadVision v2.0 &nbsp;|&nbsp; Generated "
                    f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
                    self.styles["Subtitle"],
                )
            )

            # --- Executive summary ---
            story.append(self._heading("1. Executive Summary"))
            critical = summary.by_severity.get("CRITICAL", 0)
            high = summary.by_severity.get("HIGH", 0)
            story.append(
                Paragraph(
                    f"A total of <b>{summary.total_anomalies}</b> road anomalies were "
                    f"detected, including <b>{critical}</b> critical and <b>{high}</b> "
                    f"high-severity defects. The average road condition score is "
                    f"<b>{summary.avg_road_score:.0f}/100</b>.",
                    self.styles["Normal"],
                )
            )
            story.append(Spacer(1, 0.4 * cm))

            # --- Statistics ---
            story.append(self._heading("2. Detection Statistics"))
            story.append(self._stats_table(summary))
            story.append(Spacer(1, 0.4 * cm))

            # --- Severity breakdown ---
            story.append(self._heading("3. Severity Breakdown"))
            story.append(self._severity_table(summary))
            story.append(Spacer(1, 0.4 * cm))

            # --- Gallery ---
            if gallery_images:
                story.append(self._heading("4. Annotated Image Gallery"))
                for img_bytes in gallery_images[: self.max_gallery]:
                    try:
                        story.append(
                            RLImage(io.BytesIO(img_bytes), width=15 * cm, height=9 * cm)
                        )
                        story.append(Spacer(1, 0.3 * cm))
                    except Exception:  # pragma: no cover - bad image bytes
                        continue

            # --- Geospatial ---
            if geo_summary:
                story.append(self._heading("5. Geospatial Summary"))
                geo_rows = [["Field", "Value"]] + [
                    [str(k), str(v)] for k, v in geo_summary.items()
                ]
                geo_table = Table(geo_rows, colWidths=[8 * cm, 6 * cm])
                geo_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ]
                    )
                )
                story.append(geo_table)
                story.append(Spacer(1, 0.4 * cm))

            # --- Recommendations ---
            story.append(self._heading("6. Maintenance Recommendations"))
            for rec in _recommendation_for(summary):
                story.append(Paragraph(f"&bull; {rec}", self.styles["Normal"]))
                story.append(Spacer(1, 0.15 * cm))

            # --- Priority queue ---
            if summary.repair_priority_queue:
                story.append(self._heading("7. Repair Priority Queue"))
                queue_rows = [["#", "Class", "Severity", "Urgency", "Score"]]
                for i, item in enumerate(summary.repair_priority_queue, start=1):
                    queue_rows.append(
                        [
                            str(i),
                            str(item.get("class_name", "")),
                            str(item.get("severity_level", "")),
                            str(item.get("urgency", "")),
                            f"{item.get('severity_score', 0):.2f}",
                        ]
                    )
                qtable = Table(queue_rows, repeatRows=1)
                qtable.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                story.append(qtable)

            # --- Technical metadata ---
            story.append(self._heading("8. Technical Metadata"))
            story.append(
                Paragraph(
                    "Model: YOLOv11 + SAM2 + MiDaS v3.1 &nbsp;|&nbsp; "
                    "Engine: SmartRoadVision 2.0",
                    self.styles["Normal"],
                )
            )

            doc.build(story)
            logger.info("report_generated", path=str(out_path))
            return out_path
        except Exception as exc:
            raise ReportGenerationError(f"Failed to generate PDF report: {exc}") from exc
