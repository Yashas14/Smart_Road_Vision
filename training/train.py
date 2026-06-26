"""Fine-tune YOLOv11 on the Roboflow Pothole Dataset v2.

Usage::

    python training/train.py --epochs 100 --batch 16 --imgsz 640

Requires the ``training`` extra: ``pip install .[training]``. Set
``ROBOFLOW_API_KEY`` (and optionally ``WANDB_API_KEY``) in the environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from training.dataset_prep import prepare_dataset

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line training arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 for road anomalies")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="smartroad_yolov11")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    """Run YOLOv11 fine-tuning and optionally export to ONNX.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the best checkpoint produced by training.
    """
    from ultralytics import YOLO

    settings = get_settings()
    data_yaml = args.data or str(prepare_dataset())

    # Enable Weights & Biases logging if an API key is present.
    if settings.wandb_api_key:
        try:
            import wandb

            wandb.login(key=settings.wandb_api_key)
            logger.info("wandb_enabled")
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb_init_failed", error=str(exc))

    model = YOLO(args.weights)
    logger.info("training_start", data=data_yaml, epochs=args.epochs)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        augment=True,
        amp=True,
        val=True,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("training_complete", best=str(best))

    if args.export_onnx and best.exists():
        export_model = YOLO(str(best))
        onnx_path = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=True)
        logger.info("onnx_exported", path=str(onnx_path))

    return best


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
