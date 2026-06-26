"""Maintenance cost estimation for detected road anomalies.

Turns raw anomaly detections into actionable budget figures by combining a
per-class base repair cost with multipliers for severity and physical size.
The estimator is deterministic and dependency-free so it is easy to unit-test
and to embed inside reports, dashboards or work-order generation.

Cost model::

    cost = base_cost[class]
         * (1 + severity_multiplier * severity_score)
         * area_factor

where ``area_factor`` grows with the real-world area (``area_m2``) when
available, otherwise falls back to a pixel-area proxy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.detection.types import AnomalyDetection

# Indicative base repair costs per anomaly class (currency-agnostic units).
_DEFAULT_BASE_COSTS: dict[str, float] = {
    "pothole": 120.0,
    "road_degradation": 90.0,
    "crack": 45.0,
    "hump": 60.0,
}

_DEFAULT_FALLBACK_COST = 50.0


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Cost breakdown for a single anomaly."""

    class_name: str
    severity_level: str
    urgency: str
    base_cost: float
    severity_factor: float
    area_factor: float
    estimated_cost: float
    currency: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "severity_level": self.severity_level,
            "urgency": self.urgency,
            "base_cost": self.base_cost,
            "severity_factor": self.severity_factor,
            "area_factor": self.area_factor,
            "estimated_cost": self.estimated_cost,
            "currency": self.currency,
        }


@dataclass(slots=True)
class CostReport:
    """Aggregated cost report across many anomalies."""

    total_cost: float = 0.0
    currency: str = "USD"
    by_class: dict[str, float] = field(default_factory=dict)
    by_urgency: dict[str, float] = field(default_factory=dict)
    items: list[CostEstimate] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost": self.total_cost,
            "currency": self.currency,
            "count": self.count,
            "by_class": self.by_class,
            "by_urgency": self.by_urgency,
            "items": [item.to_dict() for item in self.items],
        }


class MaintenanceCostEstimator:
    """Estimate repair costs for road anomalies.

    Args:
        base_costs: Optional override of per-class base costs.
        currency: Currency label attached to estimates.
        severity_multiplier: Scales how strongly severity inflates the cost.
        area_reference_m2: Real-world area (m²) that doubles the area factor.
        fallback_cost: Base cost used for unknown anomaly classes.
    """

    def __init__(
        self,
        base_costs: dict[str, float] | None = None,
        currency: str = "USD",
        severity_multiplier: float = 1.5,
        area_reference_m2: float = 1.0,
        fallback_cost: float = _DEFAULT_FALLBACK_COST,
    ) -> None:
        self.base_costs = base_costs or dict(_DEFAULT_BASE_COSTS)
        self.currency = currency
        self.severity_multiplier = max(0.0, severity_multiplier)
        self.area_reference_m2 = max(1e-6, area_reference_m2)
        self.fallback_cost = max(0.0, fallback_cost)

    def estimate_one(self, det: AnomalyDetection) -> CostEstimate:
        """Estimate the repair cost for a single detection.

        Args:
            det: A scored :class:`AnomalyDetection`.

        Returns:
            A :class:`CostEstimate` for the anomaly.
        """
        base = float(self.base_costs.get(det.class_name, self.fallback_cost))
        severity_score = max(0.0, min(1.0, float(det.severity_score)))
        severity_factor = round(1.0 + self.severity_multiplier * severity_score, 4)

        if det.area_m2 is not None and det.area_m2 > 0:
            area_factor = round(1.0 + det.area_m2 / self.area_reference_m2, 4)
        else:
            area_factor = 1.0

        estimated = round(base * severity_factor * area_factor, 2)
        return CostEstimate(
            class_name=det.class_name,
            severity_level=str(det.severity_level),
            urgency=str(det.urgency),
            base_cost=base,
            severity_factor=severity_factor,
            area_factor=area_factor,
            estimated_cost=estimated,
            currency=self.currency,
        )

    def estimate(self, detections: Iterable[AnomalyDetection]) -> CostReport:
        """Estimate the total repair cost across many detections.

        Args:
            detections: Scored detections to cost.

        Returns:
            An aggregated :class:`CostReport` with per-class and per-urgency
            breakdowns and a sorted item list (most expensive first).
        """
        report = CostReport(currency=self.currency)
        for det in detections:
            est = self.estimate_one(det)
            report.items.append(est)
            report.total_cost += est.estimated_cost
            report.by_class[est.class_name] = round(
                report.by_class.get(est.class_name, 0.0) + est.estimated_cost, 2
            )
            report.by_urgency[est.urgency] = round(
                report.by_urgency.get(est.urgency, 0.0) + est.estimated_cost, 2
            )

        report.total_cost = round(report.total_cost, 2)
        report.items.sort(key=lambda e: e.estimated_cost, reverse=True)
        return report
