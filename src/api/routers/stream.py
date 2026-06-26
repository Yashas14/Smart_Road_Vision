"""WebSocket live-stream endpoint for camera/RTSP feeds.

Streams annotated JPEG frames plus a JSON metadata sidecar to the connected
client. Frames are produced by :class:`StreamPipeline` and throttled to the
configured maximum FPS.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_image_pipeline, ws_manager
from src.core.logging import get_logger
from src.pipeline.stream_pipeline import StreamPipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["stream"])


@router.websocket("/stream")
async def stream_endpoint(
    websocket: WebSocket,
    source: str = Query("webcam", description="webcam | device index | RTSP/HTTP URL"),
) -> None:
    """Stream annotated detection frames over a WebSocket.

    For each processed frame the server sends a JSON metadata message followed
    by the binary JPEG payload. The client closes the socket to stop streaming.
    """
    await ws_manager.connect(websocket)
    stop = asyncio.Event()
    pipeline = StreamPipeline(pipeline=get_image_pipeline())

    try:
        async for frame in pipeline.stream(source, stop=stop):
            await websocket.send_json(
                {
                    "type": "meta",
                    "frame_index": frame.frame_index,
                    "fps": frame.fps,
                    "count": frame.result.count,
                    "road_condition_score": frame.result.road_condition_score,
                    "density_alert": frame.density_alert,
                    "detections": [d.to_dict() for d in frame.result.detections],
                }
            )
            await websocket.send_bytes(frame.jpeg_bytes)
    except WebSocketDisconnect:
        logger.info("ws_stream_disconnected", source=source)
    except Exception as exc:  # pragma: no cover
        logger.error("ws_stream_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        stop.set()
        await ws_manager.disconnect(websocket)
