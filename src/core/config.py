"""Application configuration via Pydantic settings.

All configuration is loaded from environment variables (and an optional ``.env``
file). YAML config files in ``configs/`` provide structured defaults that can be
merged in by :func:`load_yaml_config` when richer, nested settings are required.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment.

    Attributes are grouped by concern: application, logging, models, severity
    weights, database, redis/celery, reporting and streaming.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "SmartRoadVision"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_secret_key: str = "change-me"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:8501"

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False

    # --- Models ---
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    yolo_weights: Path = Field(default=PROJECT_ROOT / "models" / "yolov11-pothole.pt")
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_device: str = "auto"
    yolo_half: bool = True
    sam2_checkpoint: Path = Field(default=PROJECT_ROOT / "models" / "sam2_hiera_small.pt")
    sam2_config: str = "sam2_hiera_s.yaml"
    midas_model_type: str = "DPT_Hybrid"
    enable_segmentation: bool = True
    enable_depth: bool = True

    # --- Severity weights ---
    severity_w_area: float = 0.30
    severity_w_depth: float = 0.35
    severity_w_confidence: float = 0.15
    severity_w_class: float = 0.20

    # --- Database ---
    database_url: str = (
        "postgresql+asyncpg://smartroad:smartroad_secret@localhost:5432/smartroadvision"
    )

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- Reporting ---
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")
    max_gallery_images: int = 20

    # --- Streaming ---
    stream_max_fps: int = 30
    stream_density_window: int = 30
    stream_density_alert_threshold: int = 5

    # --- Training (optional) ---
    roboflow_api_key: str = ""
    wandb_api_key: str = ""

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, value: str) -> str:
        return value.strip()

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list of URLs."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        """True when running in a production-like environment."""
        return self.app_env.lower() in {"production", "prod", "staging"}

    @property
    def severity_weights(self) -> dict[str, float]:
        """Severity scoring weights as a mapping."""
        return {
            "area": self.severity_w_area,
            "depth": self.severity_w_depth,
            "confidence": self.severity_w_confidence,
            "class": self.severity_w_class,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Returns:
        The singleton settings object for the running process.
    """
    return Settings()


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file from the ``configs`` directory.

    Args:
        name: File name (e.g. ``"model_config.yaml"``).

    Returns:
        Parsed YAML content as a dictionary. Empty dict if the file is absent.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
