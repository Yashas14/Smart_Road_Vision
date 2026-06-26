"""Latency/throughput benchmark for the detection pipeline.

Runs N inference iterations on a synthetic or provided image and reports mean,
median and p95 latency plus throughput (FPS).

Usage::

    python scripts/benchmark.py --iterations 100
    python scripts/benchmark.py --image data/samples/road.jpg --iterations 200
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from src.core.logging import configure_logging, get_logger
from src.pipeline.image_pipeline import ImagePipeline

logger = get_logger(__name__)


def _load_image(path: str | None, size: int) -> np.ndarray:
    """Load the benchmark image or synthesise one of the given size."""
    if path and Path(path).exists():
        img = cv2.imread(path)
        if img is not None:
            return img
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def benchmark(image: np.ndarray, iterations: int, warmup: int = 5) -> dict[str, float]:
    """Benchmark the image pipeline.

    Args:
        image: Input image to repeatedly process.
        iterations: Number of timed iterations.
        warmup: Number of untimed warm-up iterations.

    Returns:
        A dict of latency statistics and throughput.
    """
    pipeline = ImagePipeline()
    pipeline.warmup()

    for _ in range(warmup):
        pipeline.process(image, annotate=False)

    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        pipeline.process(image, annotate=False)
        latencies.append((time.perf_counter() - start) * 1000.0)

    latencies.sort()
    mean_ms = statistics.fmean(latencies)
    return {
        "iterations": iterations,
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[int(0.95 * len(latencies)) - 1], 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
        "fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
    }


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Benchmark the detection pipeline")
    parser.add_argument("--image", default=None, help="optional input image path")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    image = _load_image(args.image, args.size)
    results = benchmark(image, args.iterations)

    print("\n=== Benchmark Results ===")
    for key, value in results.items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
