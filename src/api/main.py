"""FastAPI application factory with lifespan, middleware and CORS.

Loads the detection pipeline at startup (model warm-up), registers all routers
under the configured API prefix, installs a structured exception handler for
domain errors, and handles graceful shutdown of background resources.
"""

from __future__ import annotations

import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import get_redis, set_image_pipeline
from src.api.routers import analytics, detection, health, reports, stream
from src.core.config import get_settings
from src.core.exceptions import SmartRoadVisionError
from src.core.logging import configure_logging, get_logger
from src.pipeline.image_pipeline import ImagePipeline

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown: warm up models, init DB, dispose resources."""
    logger.info("startup_begin", env=settings.app_env)

    pipeline = ImagePipeline(settings)
    try:
        pipeline.warmup()
    except Exception as exc:  # pragma: no cover - model may be absent in CI
        logger.warning("pipeline_warmup_skipped", error=str(exc))
    set_image_pipeline(pipeline)

    # Best-effort DB initialisation (no-op if DB unavailable in dev/CI).
    try:
        from src.database.models import init_db

        await init_db()
    except Exception as exc:  # pragma: no cover
        logger.warning("db_init_skipped", error=str(exc))

    await get_redis()

    # Initialise the offline detection store (history/analytics/map).
    try:
        from src.storage import get_store

        get_store()
    except Exception as exc:  # pragma: no cover
        logger.warning("store_init_skipped", error=str(exc))

    def _handle_sigterm(*_: object) -> None:
        logger.info("sigterm_received")

    with suppress(ValueError):  # pragma: no cover - not in main thread
        signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info("startup_complete")
    yield
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        The configured :class:`FastAPI` instance.
    """
    app = FastAPI(
        title="SmartRoadVision API",
        description="Intelligent road surface anomaly detection system.",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SmartRoadVisionError)
    async def _domain_error_handler(_: Request, exc: SmartRoadVisionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    prefix = settings.api_v1_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(detection.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(stream.router, prefix=prefix)
    app.include_router(reports.router, prefix=prefix)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {
            "service": "SmartRoadVision",
            "version": "2.0.0",
            "frontend": "/app",
            "docs": "/docs",
            "health": f"{prefix}/health",
        }

    # Serve the web frontend (single-page app) at /app.
    if FRONTEND_DIR.exists():
        app.mount(
            "/app",
            StaticFiles(directory=str(FRONTEND_DIR), html=True),
            name="frontend",
        )

    return app


app = create_app()
