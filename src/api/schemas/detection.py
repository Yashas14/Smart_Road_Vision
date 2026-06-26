"""Detection request/response schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.detection.types import AnomalyDetection, FrameResult


class GeoCoordinate(BaseModel):
    """A WGS84 coordinate supplied with a request or extracted from EXIF."""

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    altitude: float | None = None


class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class AnomalyResult(BaseModel):
    """A single detected anomaly returned by the API."""

    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: BBox
    bbox_xywhn: list[float] = Field(..., min_length=4, max_length=4)
    polygon_mask: list[list[float]] | None = None
    severity_score: float = Field(..., ge=0.0, le=1.0)
    severity_level: str
    urgency: str
    depth_mm: float | None = None
    area_px: float
    area_m2: float | None = None
    track_id: int | None = None
    timestamp: datetime

    @classmethod
    def from_detection(
        cls, det: AnomalyDetection, image_width: int, image_height: int
    ) -> AnomalyResult:
        """Build a response model from an internal detection.

        Args:
            det: The internal detection dataclass.
            image_width: Source image width for normalisation.
            image_height: Source image height for normalisation.

        Returns:
            A populated :class:`AnomalyResult`.
        """
        x1, y1, x2, y2 = det.bbox.xyxy()
        cx, cy, w, h = det.bbox.xywhn(image_width, image_height)
        return cls(
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox_xyxy=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            bbox_xywhn=[cx, cy, w, h],
            polygon_mask=([[px, py] for px, py in det.polygon_mask] if det.polygon_mask else None),
            severity_score=det.severity_score,
            severity_level=str(det.severity_level),
            urgency=str(det.urgency),
            depth_mm=det.depth_mm,
            area_px=det.area_px,
            area_m2=det.area_m2,
            track_id=det.track_id,
            timestamp=det.timestamp,
        )


class DetectionResponse(BaseModel):
    """Response payload for a single-image detection request."""

    detection_id: int | None = None
    detections: list[AnomalyResult]
    count: int
    image_width: int
    image_height: int
    road_condition_score: float
    annotated_image_base64: str | None = None
    processing_time_ms: float
    model_version: str
    location: GeoCoordinate | None = None
    estimated_repair_cost: float | None = None
    currency: str | None = None

    @classmethod
    def from_frame_result(
        cls,
        result: FrameResult,
        annotated_image_base64: str | None = None,
        detection_id: int | None = None,
        location: GeoCoordinate | None = None,
        estimated_repair_cost: float | None = None,
        currency: str | None = None,
    ) -> DetectionResponse:
        """Build the response from an internal :class:`FrameResult`."""
        return cls(
            detection_id=detection_id,
            detections=[
                AnomalyResult.from_detection(d, result.image_width, result.image_height)
                for d in result.detections
            ],
            count=result.count,
            image_width=result.image_width,
            image_height=result.image_height,
            road_condition_score=result.road_condition_score or 100.0,
            annotated_image_base64=annotated_image_base64,
            processing_time_ms=result.processing_time_ms,
            model_version=result.model_version,
            location=location,
            estimated_repair_cost=estimated_repair_cost,
            currency=currency,
        )


class BatchItemResult(BaseModel):
    """Per-image outcome within a batch detection request."""

    filename: str
    detection_id: int | None = None
    count: int = 0
    road_condition_score: float | None = None
    estimated_repair_cost: float | None = None
    annotated_image_base64: str | None = None
    error: str | None = None


class BatchDetectionResponse(BaseModel):
    """Aggregate response for a batch of images."""

    total_images: int
    succeeded: int
    failed: int
    total_anomalies: int
    avg_road_score: float
    total_repair_cost: float
    currency: str
    items: list[BatchItemResult]


class VideoFrameSample(BaseModel):
    """A sampled annotated frame from synchronous video processing."""

    frame_index: int
    count: int
    road_condition_score: float | None = None
    annotated_image_base64: str


class VideoSyncResponse(BaseModel):
    """Result of synchronous (inline) video processing."""

    total_frames: int
    processed_frames: int
    total_detections: int
    unique_anomalies: int
    avg_road_score: float
    per_class_counts: dict[str, int]
    estimated_repair_cost: float
    currency: str
    processing_time_ms: float
    model_version: str
    sample_frames: list[VideoFrameSample]


class VideoTaskResponse(BaseModel):
    """Response when a video is submitted for async processing."""

    task_id: str
    status: str = "PENDING"


class VideoResultResponse(BaseModel):
    """Polled result for an async video processing task."""

    task_id: str
    status: str
    progress: dict[str, int | str] | None = None
    result: dict[str, object] | None = None


class DetectionListItem(BaseModel):
    """Compact detection record for history listings."""

    id: int
    source: str
    anomaly_count: int
    road_condition_score: float
    created_at: datetime
    latitude: float | None = None
    longitude: float | None = None


class HealthResponse(BaseModel):
    """Liveness/readiness response."""

    status: str = "ok"
    version: str
    model_loaded: bool
    environment: str
    using_fallback: bool = False
    engine: str = ""
