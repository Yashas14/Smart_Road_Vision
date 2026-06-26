"""Evaluate a trained YOLOv11 model: mAP, precision, recall and F1.

Usage::

    python training/evaluate.py --weights runs/train/smartroad_yolov11/weights/best.pt \
        --data data/datasets/pothole-detection-v2/data.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

from src.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv11 model")
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run validation and return key metrics.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A dict with mAP@50, mAP@50-95, precision, recall and F1.
    """
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        verbose=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    logger.info("evaluation_complete", **results)
    return results


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    results = evaluate(args)
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
