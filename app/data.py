from __future__ import annotations

import cv2
import numpy as np

from .depth import _scene


def demo_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, truth_m = _scene(42, 256)
    truth = truth_m * 1000.0
    small = cv2.resize(truth, (64, 64), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(42)
    sparse = small + rng.normal(0, 12, small.shape).astype(np.float32)
    sparse[rng.random(sparse.shape) < 0.06] = 0
    return rgb, sparse.astype(np.float32), truth.astype(np.float32)


def benchmark_scene(
    seed: int,
    size: int = 128,
    scale: int = 4,
    noise_std_mm: float = 18.0,
    missing_rate: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scale not in (2, 4, 8):
        raise ValueError("Benchmark scale must be 2, 4, or 8.")
    if noise_std_mm < 0 or not 0 <= missing_rate < 1:
        raise ValueError("Noise standard deviation and missing rate must be non-negative and below 100%.")
    rgb, truth_m = _scene(seed, size)
    truth = truth_m * 1000.0
    low_size = max(8, size // scale)
    sparse = cv2.resize(truth, (low_size, low_size), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed + 11)
    sparse += rng.normal(0, noise_std_mm, sparse.shape).astype(np.float32)
    sparse[rng.random(sparse.shape) < missing_rate] = 0
    return rgb, np.maximum(sparse, 0).astype(np.float32), truth.astype(np.float32)
