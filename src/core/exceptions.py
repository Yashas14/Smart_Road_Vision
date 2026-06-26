"""Custom exception hierarchy for SmartRoadVision.

All application errors derive from :class:`SmartRoadVisionError`, which makes it
trivial to catch every domain error at the API boundary and convert it into a
structured HTTP response.
"""

from __future__ import annotations


class SmartRoadVisionError(Exception):
    """Base class for all SmartRoadVision domain errors.

    Args:
        message: Human-readable error description.
        status_code: Suggested HTTP status code for API responses.
    """

    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, str | int]:
        """Serialise the error for an API response body."""
        return {"error": self.__class__.__name__, "detail": self.message}


class ModelLoadError(SmartRoadVisionError):
    """Raised when a model checkpoint cannot be loaded or initialised."""

    status_code = 503


class DetectionError(SmartRoadVisionError):
    """Raised when inference fails for a frame or image."""

    status_code = 422


class PreprocessingError(SmartRoadVisionError):
    """Raised when image preprocessing fails."""

    status_code = 422


class ValidationError(SmartRoadVisionError):
    """Raised when input validation fails outside of Pydantic."""

    status_code = 400


class GeoExtractionError(SmartRoadVisionError):
    """Raised when GPS/EXIF extraction fails irrecoverably."""

    status_code = 422


class ReportGenerationError(SmartRoadVisionError):
    """Raised when a PDF report cannot be generated."""

    status_code = 500


class DatabaseError(SmartRoadVisionError):
    """Raised for database connectivity or query failures."""

    status_code = 503


class StreamError(SmartRoadVisionError):
    """Raised when a video stream source cannot be opened or read."""

    status_code = 422
