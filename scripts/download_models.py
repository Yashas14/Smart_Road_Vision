"""Download required model weights: YOLOv11, SAM2 and MiDaS.

Usage::

    python scripts/download_models.py --all
    python scripts/download_models.py --yolo --midas

YOLOv11 base weights are fetched by ultralytics on first use; this script
pre-fetches them. SAM2 and MiDaS are downloaded from their official releases.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
)


def _download(url: str, dest: Path) -> None:
    """Download a file with a simple progress log."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("already_downloaded", path=str(dest))
        return
    logger.info("downloading", url=url, dest=str(dest))
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted official URL
    logger.info("downloaded", path=str(dest), size_mb=round(dest.stat().st_size / 1e6, 1))


def download_yolo(models_dir: Path) -> None:
    """Pre-fetch a base YOLOv11 checkpoint via ultralytics."""
    try:
        from ultralytics import YOLO

        YOLO("yolo11n.pt")  # triggers download into ultralytics cache
        logger.info("yolo_base_ready")
    except Exception as exc:  # pragma: no cover
        logger.warning("yolo_download_failed", error=str(exc))


def download_sam2(models_dir: Path) -> None:
    """Download the SAM2 hiera-small checkpoint."""
    try:
        _download(SAM2_URL, models_dir / "sam2_hiera_small.pt")
    except Exception as exc:  # pragma: no cover
        logger.warning("sam2_download_failed", error=str(exc))


def download_midas() -> None:
    """Trigger MiDaS download via torch.hub cache."""
    try:
        import torch

        settings = get_settings()
        torch.hub.load("intel-isl/MiDaS", settings.midas_model_type, trust_repo=True)
        torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        logger.info("midas_ready", model=settings.midas_model_type)
    except Exception as exc:  # pragma: no cover
        logger.warning("midas_download_failed", error=str(exc))


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Download SmartRoadVision models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--midas", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.yolo:
        download_yolo(models_dir)
    if args.all or args.sam2:
        download_sam2(models_dir)
    if args.all or args.midas:
        download_midas()

    if not any([args.all, args.yolo, args.sam2, args.midas]):
        parser.print_help()


if __name__ == "__main__":
    main()
