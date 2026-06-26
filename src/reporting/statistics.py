"""Pandas-based aggregation and trend analysis over detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class StatisticsSummary:
    """Aggregate statistics computed over a set of anomaly records."""

    total_anomalies: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_severity_score: float = 0.0
    avg_road_score: float = 100.0
    repair_priority_queue: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_anomalies": self.total_anomalies,
            "by_class": self.by_class,
            "by_severity": self.by_severity,
            "avg_confidence": self.avg_confidence,
            "avg_severity_score": self.avg_severity_score,
            "avg_road_score": self.avg_road_score,
            "repair_priority_queue": self.repair_priority_queue,
        }


_URGENCY_RANK = {
    "IMMEDIATE": 0,
    "URGENT": 1,
    "SCHEDULE_REPAIR": 2,
    "MONITOR": 3,
}


class DetectionStatistics:
    """Compute statistics and time-series trends from anomaly records.

    Each record is expected to be a mapping with keys such as ``class_name``,
    ``severity_level``, ``severity_score``, ``confidence``, ``urgency`` and an
    optional ``created_at`` timestamp and ``road_condition_score``.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.df = pd.DataFrame(records)

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    def summary(self, road_scores: list[float] | None = None) -> StatisticsSummary:
        """Compute an aggregate summary.

        Args:
            road_scores: Optional per-frame road condition scores to average.

        Returns:
            A :class:`StatisticsSummary`.
        """
        if self.is_empty:
            return StatisticsSummary(
                avg_road_score=(
                    round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0
                )
            )

        by_class = self.df["class_name"].value_counts().to_dict()
        by_severity = self.df["severity_level"].value_counts().to_dict()
        avg_conf = round(float(self.df["confidence"].mean()), 4)
        avg_sev = round(float(self.df["severity_score"].mean()), 4)
        avg_road = round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0

        queue = self._priority_queue()
        return StatisticsSummary(
            total_anomalies=len(self.df),
            by_class={str(k): int(v) for k, v in by_class.items()},
            by_severity={str(k): int(v) for k, v in by_severity.items()},
            avg_confidence=avg_conf,
            avg_severity_score=avg_sev,
            avg_road_score=avg_road,
            repair_priority_queue=queue,
        )

    def _priority_queue(self, top_n: int = 20) -> list[dict[str, Any]]:
        """Rank anomalies by urgency then severity for a repair queue."""
        df = self.df.copy()
        if "urgency" in df.columns:
            df["_rank"] = df["urgency"].map(_URGENCY_RANK).fillna(9)
        else:
            df["_rank"] = 9
        df = df.sort_values(by=["_rank", "severity_score"], ascending=[True, False]).head(top_n)
        cols = [
            c
            for c in ["class_name", "severity_level", "urgency", "severity_score", "depth_mm"]
            if c in df.columns
        ]
        return df[cols].to_dict(orient="records")

    def hourly_trend(self) -> pd.DataFrame:
        """Return detections-per-hour as a DataFrame (empty if no timestamps)."""
        if self.is_empty or "created_at" not in self.df.columns:
            return pd.DataFrame(columns=["period", "count"])
        df = self.df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = df.dropna(subset=["created_at"])
        if df.empty:
            return pd.DataFrame(columns=["period", "count"])
        grouped = df.set_index("created_at").resample("1h").size().reset_index(name="count")
        grouped = grouped.rename(columns={"created_at": "period"})
        return grouped

    def rolling_severity(self, window: int = 10) -> pd.DataFrame:
        """Return a rolling-average severity series for trend charts."""
        if self.is_empty or "severity_score" not in self.df.columns:
            return pd.DataFrame(columns=["index", "rolling_severity"])
        series = self.df["severity_score"].rolling(window, min_periods=1).mean()
        return pd.DataFrame({"index": range(len(series)), "rolling_severity": series.to_numpy()})
