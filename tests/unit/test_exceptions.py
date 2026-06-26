"""Unit tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest

from src.core.exceptions import (
    DatabaseError,
    DetectionError,
    GeoExtractionError,
    ModelLoadError,
    PreprocessingError,
    ReportGenerationError,
    SmartRoadVisionError,
    StreamError,
    ValidationError,
)

ALL_ERRORS = [
    (ModelLoadError, 503),
    (DetectionError, 422),
    (PreprocessingError, 422),
    (ValidationError, 400),
    (GeoExtractionError, 422),
    (ReportGenerationError, 500),
    (DatabaseError, 503),
    (StreamError, 422),
]


@pytest.mark.parametrize("exc_cls, expected_status", ALL_ERRORS)
def test_default_status_codes(exc_cls: type[SmartRoadVisionError], expected_status: int) -> None:
    err = exc_cls("boom")
    assert err.status_code == expected_status


@pytest.mark.parametrize("exc_cls, _status", ALL_ERRORS)
def test_all_errors_subclass_base(exc_cls: type[SmartRoadVisionError], _status: int) -> None:
    assert issubclass(exc_cls, SmartRoadVisionError)
    assert isinstance(exc_cls("x"), Exception)


def test_message_is_preserved() -> None:
    err = DetectionError("inference failed")
    assert err.message == "inference failed"
    assert str(err) == "inference failed"


def test_to_dict_shape() -> None:
    err = ValidationError("bad input")
    payload = err.to_dict()
    assert payload == {"error": "ValidationError", "detail": "bad input"}


def test_status_code_override() -> None:
    err = SmartRoadVisionError("teapot", status_code=418)
    assert err.status_code == 418


def test_base_default_status_is_500() -> None:
    assert SmartRoadVisionError("generic").status_code == 500


def test_can_be_raised_and_caught_via_base() -> None:
    with pytest.raises(SmartRoadVisionError):
        raise StreamError("camera lost")
