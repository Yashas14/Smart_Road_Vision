"""Detection endpoints: image, video (async), and history listing."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import anyio
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.analytics.cost_estimator import MaintenanceCostEstimator
from src.api.dependencies import get_db, get_image_pipeline
from src.api.schemas.detection import (
    BatchDetectionResponse,
    BatchItemResult,
    DetectionListItem,
    DetectionResponse,
    GeoCoordinate,
    VideoFrameSample,
    VideoResultResponse,
    VideoSyncResponse,
    VideoTaskResponse,
)
from src.core.exceptions import SmartRoadVisionError
from src.core.logging import get_logger
from src.database import crud
from src.detection.types import FrameResult
from src.pipeline.image_pipeline import ImagePipeline
from src.storage import get_store
from src.utils.geo_utils import GeoPoint, extract_gps_from_exif
from src.utils.image_utils import bytes_to_image, image_to_base64, resize_keep_aspect

logger = get_logger(__name__)
router = APIRouter(prefix="/detect", tags=["detection"])
_cost_estimator = MaintenanceCostEstimator()


def _thumbnail(image: np.ndarray | None, max_side: int = 360) -> str | None:
    """Create a small base64 JPEG thumbnail for history display."""
    if image is None:
        return None
    try:
        return image_to_base64(resize_keep_aspect(image, max_side=max_side), ext=".jpg")
    except Exception:  # pragma: no cover - defensive
        return None


def _persist_to_store(
    result: FrameResult,
    source: str,
    point: GeoPoint | None,
    cost_total: float,
    currency: str,
    thumbnail: str | None,
) -> int | None:
    """Best-effort persistence to the offline SQLite store."""
    try:
        return get_store().save_detection(
            result,
            source=source,
            latitude=point.latitude if point else None,
            longitude=point.longitude if point else None,
            estimated_repair_cost=cost_total,
            currency=currency,
            thumbnail=thumbnail,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("store_persist_failed", error=str(exc))
        return None


@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    file: Annotated[UploadFile, File(description="Road image to analyse")],
    pipeline: Annotated[ImagePipeline, Depends(get_image_pipeline)],
    session: Annotated[AsyncSession, Depends(get_db)],
    latitude: Annotated[float | None, Form()] = None,
    longitude: Annotated[float | None, Form()] = None,
    annotate: Annotated[bool, Form()] = True,
    persist: Annotated[bool, Form()] = True,
) -> DetectionResponse:
    """Detect anomalies in an uploaded image.

    Accepts a multipart image upload plus optional GPS coordinates. If no
    coordinates are supplied, GPS is extracted from EXIF when present.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        image = bytes_to_image(raw)
    except SmartRoadVisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    # Resolve GPS: explicit params take precedence over EXIF.
    point: GeoPoint | None = None
    if latitude is not None and longitude is not None:
        point = GeoPoint(latitude=latitude, longitude=longitude)
    else:
        point = extract_gps_from_exif(raw)

    try:
        result, annotated_img = pipeline.process(image, annotate=annotate)
    except SmartRoadVisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    detection_id: int | None = None
    if persist:
        try:
            record = await crud.save_frame_result(session, result, source="image", point=point)
            detection_id = record.id
        except Exception as exc:  # pragma: no cover - db optional in some envs
            logger.warning("persist_failed", error=str(exc))

    annotated_b64 = (
        image_to_base64(annotated_img) if annotate and annotated_img is not None else None
    )
    location = GeoCoordinate(latitude=point.latitude, longitude=point.longitude) if point else None
    cost_report = _cost_estimator.estimate(result.detections)
    store_id = None
    if persist:
        thumb = _thumbnail(annotated_img if annotated_img is not None else image)
        store_id = _persist_to_store(
            result, "image", point, cost_report.total_cost, cost_report.currency, thumb
        )
    if detection_id is None:
        detection_id = store_id
    return DetectionResponse.from_frame_result(
        result,
        annotated_image_base64=annotated_b64,
        detection_id=detection_id,
        location=location,
        estimated_repair_cost=cost_report.total_cost,
        currency=cost_report.currency,
    )


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: Annotated[list[UploadFile], File(description="Road images to analyse")],
    pipeline: Annotated[ImagePipeline, Depends(get_image_pipeline)],
    annotate: Annotated[bool, Form()] = True,
    persist: Annotated[bool, Form()] = True,
) -> BatchDetectionResponse:
    """Analyse multiple images in a single request."""
    items: list[BatchItemResult] = []
    total_anomalies = 0
    road_scores: list[float] = []
    total_cost = 0.0
    currency = "USD"
    succeeded = 0

    for upload in files:
        name = upload.filename or "image"
        try:
            raw = await upload.read()
            if not raw:
                raise SmartRoadVisionError("Empty file", status_code=400)
            image = bytes_to_image(raw)
            point = extract_gps_from_exif(raw)
            result, annotated_img = pipeline.process(image, annotate=annotate)
            cost = _cost_estimator.estimate(result.detections)
            currency = cost.currency
            total_cost += cost.total_cost
            total_anomalies += result.count
            if result.road_condition_score is not None:
                road_scores.append(result.road_condition_score)
            det_id = None
            if persist:
                det_id = _persist_to_store(
                    result,
                    "batch",
                    point,
                    cost.total_cost,
                    cost.currency,
                    _thumbnail(annotated_img if annotated_img is not None else image),
                )
            items.append(
                BatchItemResult(
                    filename=name,
                    detection_id=det_id,
                    count=result.count,
                    road_condition_score=result.road_condition_score,
                    estimated_repair_cost=cost.total_cost,
                    annotated_image_base64=(
                        image_to_base64(annotated_img)
                        if annotate and annotated_img is not None
                        else None
                    ),
                )
            )
            succeeded += 1
        except Exception as exc:
            logger.warning("batch_item_failed", filename=name, error=str(exc))
            items.append(BatchItemResult(filename=name, error=str(exc)))

    return BatchDetectionResponse(
        total_images=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        total_anomalies=total_anomalies,
        avg_road_score=round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0,
        total_repair_cost=round(total_cost, 2),
        currency=currency,
        items=items,
    )


@router.post("/video/sync", response_model=VideoSyncResponse)
async def detect_video_sync(
    file: Annotated[UploadFile, File(description="Road video to analyse inline")],
    max_frames: Annotated[int, Form()] = 40,
    persist: Annotated[bool, Form()] = True,
) -> VideoSyncResponse:
    """Process a short video synchronously (no Celery), sampling frames evenly.

    Returns aggregate statistics plus a handful of annotated sample frames so
    results can be previewed immediately in the browser.
    """
    from src.detection.postprocessor import draw_annotations
    from src.pipeline.video_pipeline import VideoPipeline

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot decode the uploaded video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max(1, max_frames)) if total else 5

    pipeline = VideoPipeline(sample_every=step)
    samples: list[VideoFrameSample] = []
    per_class: dict[str, int] = {}
    road_scores: list[float] = []
    track_ids: set[int] = set()
    total_detections = 0
    processed = 0
    total_cost = 0.0
    currency = "USD"
    start = datetime.now()
    model_version = "opencv-heuristic-v1"

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                result, _ = pipeline.pipeline.process(frame, annotate=False)
                pipeline._apply_tracking(result.detections, frame)
                model_version = result.model_version
                processed += 1
                total_detections += result.count
                for det in result.detections:
                    if det.track_id is not None:
                        track_ids.add(det.track_id)
                    per_class[det.class_name] = per_class.get(det.class_name, 0) + 1
                if result.road_condition_score is not None:
                    road_scores.append(result.road_condition_score)
                cost = _cost_estimator.estimate(result.detections)
                currency = cost.currency
                total_cost += cost.total_cost
                if result.count and len(samples) < 6:
                    annotated = draw_annotations(frame, result.detections)
                    samples.append(
                        VideoFrameSample(
                            frame_index=idx,
                            count=result.count,
                            road_condition_score=result.road_condition_score,
                            annotated_image_base64=image_to_base64(
                                resize_keep_aspect(annotated, 640)
                            ),
                        )
                    )
                if persist and result.count:
                    result.frame_index = idx
                    _persist_to_store(result, "video", None, cost.total_cost, cost.currency, None)
            idx += 1
    finally:
        cap.release()
        with anyio.CancelScope(shield=True):
            await anyio.Path(tmp_path).unlink(missing_ok=True)

    elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
    return VideoSyncResponse(
        total_frames=idx,
        processed_frames=processed,
        total_detections=total_detections,
        unique_anomalies=len(track_ids),
        avg_road_score=round(sum(road_scores) / len(road_scores), 1) if road_scores else 100.0,
        per_class_counts=per_class,
        estimated_repair_cost=round(total_cost, 2),
        currency=currency,
        processing_time_ms=round(elapsed_ms, 1),
        model_version=model_version,
        sample_frames=samples,
    )


@router.post("/video", response_model=VideoTaskResponse)
async def detect_video(
    file: Annotated[UploadFile, File(description="Road video to analyse")],
) -> VideoTaskResponse:
    """Submit a video for asynchronous processing via Celery.

    Returns a ``task_id`` that can be polled at ``GET /detect/video/{task_id}``.
    """
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from src.pipeline.batch_pipeline import process_video_task

        task = process_video_task.delay(tmp_path, None)
        return VideoTaskResponse(task_id=task.id, status="PENDING")
    except Exception as exc:
        logger.error("video_task_submit_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}") from exc


@router.get("/video/{task_id}", response_model=VideoResultResponse)
async def get_video_result(task_id: str) -> VideoResultResponse:
    """Poll the status/result of an async video processing task."""
    try:
        from src.pipeline.batch_pipeline import celery_app

        async_result = celery_app.AsyncResult(task_id)
        status = async_result.status
        progress: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        if status == "PROGRESS":
            progress = async_result.info
        elif status == "SUCCESS":
            result = async_result.result
        return VideoResultResponse(task_id=task_id, status=status, progress=progress, result=result)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task backend unavailable: {exc}") from exc


@router.get("", response_model=list[DetectionListItem])
async def list_detections(
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    severity: Annotated[str | None, Query()] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
) -> list[DetectionListItem]:
    """List detection history with pagination and optional filters."""
    rows = await crud.list_detections(
        session,
        limit=limit,
        offset=offset,
        severity=severity,
        date_from=date_from,
        date_to=date_to,
    )
    items: list[DetectionListItem] = []
    for d in rows:
        items.append(
            DetectionListItem(
                id=d.id,
                source=d.source,
                anomaly_count=d.anomaly_count,
                road_condition_score=d.road_condition_score,
                created_at=d.created_at,
                latitude=d.location.latitude if d.location else None,
                longitude=d.location.longitude if d.location else None,
            )
        )
    return items
