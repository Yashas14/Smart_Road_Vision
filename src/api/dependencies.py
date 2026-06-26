"""FastAPI dependency-injection providers.

Provides cached singletons for the image pipeline, Redis cache and a WebSocket
connection manager, plus the async database session dependency. Heavy models
are instantiated once at application startup and shared across requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.database.models import get_session
from src.pipeline.image_pipeline import ImagePipeline

logger = get_logger(__name__)

# Singletons populated during the app lifespan.
_image_pipeline: ImagePipeline | None = None
_redis: Any | None = None


def set_image_pipeline(pipeline: ImagePipeline) -> None:
    """Register the process-wide image pipeline (called at startup)."""
    global _image_pipeline
    _image_pipeline = pipeline


def get_image_pipeline() -> ImagePipeline:
    """Return the shared :class:`ImagePipeline`, building it on demand."""
    global _image_pipeline
    if _image_pipeline is None:
        _image_pipeline = ImagePipeline(get_settings())
    return _image_pipeline


def get_detector_status() -> bool:
    """Whether the detector model is currently loaded."""
    return _image_pipeline is not None and _image_pipeline.detector.is_loaded


def get_detector_engine_info() -> tuple[bool, str]:
    """Return ``(using_fallback, model_version)`` for the active detector."""
    if _image_pipeline is None:
        return True, "unloaded"
    detector = _image_pipeline.detector
    using_fallback = bool(getattr(detector, "using_fallback", False))
    version = getattr(detector, "model_version", "") or ""
    return using_fallback, version


async def get_redis() -> Any | None:
    """Return a shared async Redis client, or ``None`` if unavailable."""
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        _redis = aioredis.from_url(settings.redis_url, decode_responses=False)
        await _redis.ping()
        return _redis
    except Exception as exc:  # pragma: no cover - redis optional in dev
        logger.warning("redis_unavailable", error=str(exc))
        return None


async def get_db() -> AsyncGenerator:
    """Yield an async database session."""
    async for session in get_session():
        yield session


def get_app_settings() -> Settings:
    """Return application settings (DI-friendly)."""
    return get_settings()


class WebSocketManager:
    """Tracks active WebSocket clients and broadcasts frames/alerts."""

    def __init__(self) -> None:
        self._connections: set[Any] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("ws_client_connected", clients=len(self._connections))

    async def disconnect(self, websocket: Any) -> None:
        """Deregister a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("ws_client_disconnected", clients=len(self._connections))

    async def broadcast_bytes(self, data: bytes) -> None:
        """Broadcast binary data to all connected clients."""
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_bytes(data)
            except Exception:  # pragma: no cover
                await self.disconnect(ws)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected clients."""
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_json(data)
            except Exception:  # pragma: no cover
                await self.disconnect(ws)


ws_manager = WebSocketManager()
