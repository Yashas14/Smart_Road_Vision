"""Unit tests for the severity scorer."""

from __future__ import annotations

import pytest

from src.detection.severity_scorer import SeverityScorer
from src.detection.types import SeverityLevel, UrgencyTag
from tests.conftest import make_detection


@pytest.fixture
def scorer() -> SeverityScorer:
    return SeverityScorer()


def test_score_is_within_unit_interval(scorer: SeverityScorer) -> None:
    det = make_detection(confidence=0.8, box=(0, 0, 320, 240), depth_mm=120.0)
    scorer.score_one(det, image_width=640, image_height=480)
    assert 0.0 <= det.severity_score <= 1.0


def test_larger_deeper_anomaly_scores_higher(scorer: SeverityScorer) -> None:
    small = make_detection(box=(0, 0, 40, 40), depth_mm=10.0, confidence=0.5)
    large = make_detection(box=(0, 0, 400, 400), depth_mm=200.0, confidence=0.95)
    scorer.score_one(small, 640, 480)
    scorer.score_one(large, 640, 480)
    assert large.severity_score > small.severity_score


def test_classification_thresholds(scorer: SeverityScorer) -> None:
    assert scorer._classify(0.1) == SeverityLevel.LOW
    assert scorer._classify(0.45) == SeverityLevel.MEDIUM
    assert scorer._classify(0.7) == SeverityLevel.HIGH
    assert scorer._classify(0.95) == SeverityLevel.CRITICAL


def test_urgency_follows_severity(scorer: SeverityScorer) -> None:
    det = make_detection(box=(0, 0, 600, 460), depth_mm=300.0, confidence=0.99)
    scorer.score_one(det, 640, 480)
    if det.severity_level == SeverityLevel.CRITICAL:
        assert det.urgency == UrgencyTag.IMMEDIATE


def test_road_condition_score_perfect_when_empty() -> None:
    assert SeverityScorer.road_condition_score([]) == 100.0


def test_road_condition_score_decreases_with_anomalies(scorer: SeverityScorer) -> None:
    det = make_detection(box=(0, 0, 400, 400), depth_mm=200.0, confidence=0.95)
    scorer.score_one(det, 640, 480)
    score = SeverityScorer.road_condition_score([det])
    assert score < 100.0


def test_class_weight_affects_score(scorer: SeverityScorer) -> None:
    pothole = make_detection(class_name="pothole", depth_mm=0.0, confidence=0.5)
    hump = make_detection(class_name="hump", depth_mm=0.0, confidence=0.5)
    scorer.score_one(pothole, 640, 480)
    scorer.score_one(hump, 640, 480)
    assert pothole.severity_score >= hump.severity_score
