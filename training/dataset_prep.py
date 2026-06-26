"""Download and prepare the Roboflow Pothole Dataset v2 for YOLO training."""

from __future__ import annotations

from pathlib import Path

from src.core.config import PROJECT_ROOT, get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"


def prepare_dataset(
    workspace: str = "road-detection",
    project: str = "pothole-detection-v2",
    version: int = 2,
) -> Path:
    """Download the Roboflow dataset in YOLOv11 format and return ``data.yaml``.

    Args:
        workspace: Roboflow workspace slug.
        project: Roboflow project slug.
        version: Dataset version number.

    Returns:
        Path to the dataset ``data.yaml`` file.

    Raises:
        RuntimeError: If the Roboflow API key is missing.
    """
    settings = get_settings()
    if not settings.roboflow_api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Add it to your .env to download the dataset."
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = DATASET_DIR / project / "data.yaml"
    if existing.exists():
        logger.info("dataset_already_present", path=str(existing))
        return existing

    from roboflow import Roboflow

    rf = Roboflow(api_key=settings.roboflow_api_key)
    rf_project = rf.workspace(workspace).project(project)
    dataset = rf_project.version(version).download("yolov11", location=str(DATASET_DIR / project))
    data_yaml = Path(dataset.location) / "data.yaml"
    logger.info("dataset_downloaded", path=str(data_yaml))
    return data_yaml


if __name__ == "__main__":
    print(prepare_dataset())
