"""Unit tests for the maintenance cost estimator."""

from __future__ import annotations

from src.analytics.cost_estimator import (
    CostEstimate,
    CostReport,
    MaintenanceCostEstimator,
)
from src.detection.types import SeverityLevel, UrgencyTag
from tests.conftest import make_detection


def _scored(class_name: str, severity_score: float, **kwargs) -> object:
    det = make_detection(class_name=class_name, **kwargs)
    det.severity_score = severity_score
    return det


def test_estimate_one_uses_base_cost_for_known_class() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=0.0)
    det = _scored("pothole", 0.0)
    det.area_m2 = None
    est = estimator.estimate_one(det)
    assert isinstance(est, CostEstimate)
    assert est.estimated_cost == 120.0  # base * 1 * 1
    assert est.currency == "USD"


def test_unknown_class_uses_fallback_cost() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=0.0, fallback_cost=77.0)
    det = _scored("alien_crater", 0.0)
    det.area_m2 = None
    est = estimator.estimate_one(det)
    assert est.base_cost == 77.0
    assert est.estimated_cost == 77.0


def test_severity_inflates_cost() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=1.5)
    det = _scored("pothole", 1.0)
    det.area_m2 = None
    est = estimator.estimate_one(det)
    # base 120 * (1 + 1.5*1.0) = 300
    assert est.severity_factor == 2.5
    assert est.estimated_cost == 300.0


def test_area_factor_increases_cost() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=0.0, area_reference_m2=2.0)
    det = _scored("pothole", 0.0)
    det.area_m2 = 2.0
    est = estimator.estimate_one(det)
    # area_factor = 1 + 2/2 = 2 ; cost = 120 * 1 * 2
    assert est.area_factor == 2.0
    assert est.estimated_cost == 240.0


def test_severity_score_is_clamped() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=1.0)
    det = _scored("crack", 5.0)  # absurd score
    det.area_m2 = None
    est = estimator.estimate_one(det)
    # clamped to 1.0 -> factor 2.0 -> 45 * 2 = 90
    assert est.estimated_cost == 90.0


def test_estimate_aggregates_total_and_breakdowns() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=0.0)
    p = _scored("pothole", 0.0)
    p.area_m2 = None
    p.urgency = UrgencyTag.IMMEDIATE
    c = _scored("crack", 0.0)
    c.area_m2 = None
    c.urgency = UrgencyTag.MONITOR
    report = estimator.estimate([p, c])
    assert isinstance(report, CostReport)
    assert report.count == 2
    assert report.total_cost == 165.0  # 120 + 45
    assert report.by_class == {"pothole": 120.0, "crack": 45.0}
    assert report.by_urgency == {"IMMEDIATE": 120.0, "MONITOR": 45.0}


def test_estimate_items_sorted_descending() -> None:
    estimator = MaintenanceCostEstimator(severity_multiplier=0.0)
    cheap = _scored("crack", 0.0)
    cheap.area_m2 = None
    pricey = _scored("pothole", 0.0)
    pricey.area_m2 = None
    report = estimator.estimate([cheap, pricey])
    assert [i.class_name for i in report.items] == ["pothole", "crack"]


def test_empty_report() -> None:
    report = MaintenanceCostEstimator().estimate([])
    assert report.total_cost == 0.0
    assert report.count == 0
    assert report.by_class == {}


def test_cost_report_to_dict() -> None:
    det = _scored("pothole", 0.5)
    det.area_m2 = None
    det.severity_level = SeverityLevel.HIGH
    report = MaintenanceCostEstimator().estimate([det])
    payload = report.to_dict()
    assert payload["count"] == 1
    assert payload["currency"] == "USD"
    assert payload["items"][0]["class_name"] == "pothole"
