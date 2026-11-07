from __future__ import annotations

from functools import lru_cache
import threading
import time

import cv2
import numpy as np
import psutil
from scipy.ndimage import distance_transform_edt

from .model import DEFAULT_CHECKPOINT, ResidualUNet, load_checkpoint, train_checkpoint


METHODS = (
    "nearest",
    "bilinear",
    "bicubic",
    "joint_bilateral",
    "guided",
    "cnn",
    "hybrid",
)

METHOD_LABELS = {
    "nearest": "Nearest neighbour",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "joint_bilateral": "Joint bilateral",
    "guided": "Guided filter",
    "cnn": "Residual U-Net",
    "hybrid": "Hybrid (Bicubic + guided + U-Net)",
}

_MODEL_LOCK = threading.Lock()


def _filled_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("The depth image contains no positive, valid measurements.")
    if valid.all():
        return depth.copy(), valid.astype(np.float32)
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return depth[tuple(nearest)], valid.astype(np.float32)


def _denoise_and_fill(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    original_valid = np.isfinite(depth) & (depth > 0)
    filled, _ = _filled_depth(depth)
    median = cv2.medianBlur(filled, 3)
    absolute_deviation = cv2.medianBlur(np.abs(filled - median), 3)
    threshold = np.maximum(0.01 * np.abs(median), 6.0 * 1.4826 * absolute_deviation)
    threshold = np.maximum(threshold, np.finfo(np.float32).eps)
    inlier_neighbors = (np.abs(filled - median) <= threshold).astype(np.float32)
    neighbor_support = cv2.boxFilter(
        inlier_neighbors,
        ddepth=-1,
        ksize=(3, 3),
        normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )
    isolated_outliers = original_valid & (np.abs(filled - median) > threshold) & (neighbor_support >= 6)
    cleaned_valid = original_valid & ~isolated_outliers
    if not cleaned_valid.any():
        cleaned_valid = original_valid
        isolated_outliers[:] = False
    cleaned = depth.copy()
    cleaned[~cleaned_valid] = 0
    filled_cleaned, _ = _filled_depth(cleaned)
    return filled_cleaned, original_valid.astype(np.float32), cleaned_valid.astype(np.float32)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _guided_filter(guide: np.ndarray, source: np.ndarray, radius: int = 5, epsilon: float = 0.0001) -> np.ndarray:
    guide = guide.astype(np.float32)
    source = source.astype(np.float32)
    size = (radius * 2 + 1, radius * 2 + 1)
    mean_guide = cv2.boxFilter(guide, -1, size, borderType=cv2.BORDER_REFLECT)
    mean_source = cv2.boxFilter(source, -1, size, borderType=cv2.BORDER_REFLECT)
    correlation = cv2.boxFilter(guide * guide, -1, size, borderType=cv2.BORDER_REFLECT)
    covariance = cv2.boxFilter(guide * source, -1, size, borderType=cv2.BORDER_REFLECT)
    variance = correlation - mean_guide * mean_guide
    slope = (covariance - mean_guide * mean_source) / (variance + epsilon)
    intercept = mean_source - slope * mean_guide
    mean_slope = cv2.boxFilter(slope, -1, size, borderType=cv2.BORDER_REFLECT)
    mean_intercept = cv2.boxFilter(intercept, -1, size, borderType=cv2.BORDER_REFLECT)
    return mean_slope * guide + mean_intercept


def _joint_bilateral(rgb: np.ndarray, source: np.ndarray, radius: int = 3) -> np.ndarray:
    guide = rgb.astype(np.float32) / 255.0
    padded_guide = cv2.copyMakeBorder(guide, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    padded_source = cv2.copyMakeBorder(source, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    height, width = source.shape
    center_guide = padded_guide[radius : radius + height, radius : radius + width]
    weights = np.zeros_like(source, dtype=np.float32)
    values = np.zeros_like(source, dtype=np.float32)
    spatial_sigma = max(radius * 0.8, 1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor_guide = padded_guide[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            neighbor_depth = padded_source[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            color_distance = np.sum((center_guide - neighbor_guide) ** 2, axis=2)
            depth_distance = (source - neighbor_depth) ** 2
            spatial_weight = np.exp(-(dx * dx + dy * dy) / (2 * spatial_sigma**2))
            weight = (
                np.exp(-color_distance / (2 * 0.08**2))
                * np.exp(-depth_distance / (2 * 0.12**2))
                * spatial_weight
            )
            weights += weight
            values += weight * neighbor_depth
    return values / np.maximum(weights, 1e-8)


def _scene(seed: int, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    depth = (2.7 + 0.35 * xx / size + 0.18 * yy / size).astype(np.float32)
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    background = rng.integers(65, 195, size=3, dtype=np.uint8)
    rgb[:] = background

    for _ in range(int(rng.integers(2, 5))):
        color = rng.integers(35, 245, size=3, dtype=np.uint8)
        object_depth = float(rng.uniform(0.7, 2.4))
        if rng.random() < 0.55:
            x1, y1 = rng.integers(3, size // 2, size=2)
            x2 = int(rng.integers(size // 2, size - 3))
            y2 = int(rng.integers(size // 2, size - 3))
            cv2.rectangle(rgb, (int(x1), int(y1)), (x2, y2), color.tolist(), -1)
            cv2.rectangle(depth, (int(x1), int(y1)), (x2, y2), object_depth, -1)
        else:
            center = tuple(int(value) for value in rng.integers(10, size - 10, size=2))
            radius = int(rng.integers(6, max(7, size // 4)))
            cv2.circle(rgb, center, radius, color.tolist(), -1)
            cv2.circle(depth, center, radius, object_depth, -1)

    rgb += rng.normal(0, 2, rgb.shape).astype(np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth


def _sample_sparse_depth(
    target: np.ndarray,
    seed: int,
    scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target.ndim != 2 or not target.size:
        raise ValueError("Training depth target must be a non-empty single-channel image.")
    if scale not in (2, 4, 8):
        raise ValueError("Training sampling scale must be 2, 4, or 8.")
    if not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("Training depth target must contain finite, positive values.")

    pattern_rng = np.random.default_rng(seed)
    offset_y, offset_x = (int(value) for value in pattern_rng.integers(0, scale, size=2))
    sample_mask = np.zeros(target.shape, dtype=bool)
    sample_mask[offset_y::scale, offset_x::scale] = True

    sparse = np.zeros(target.shape, dtype=np.float32)
    measurements = target[sample_mask].astype(np.float32)
    noise_rng = np.random.default_rng(seed + 5000)
    noise_sigma = 0.01 + 0.005 * measurements
    sparse[sample_mask] = measurements + noise_rng.normal(0, noise_sigma).astype(np.float32)
    sparse[sample_mask] = np.maximum(sparse[sample_mask], np.finfo(np.float32).eps)

    observed_mask = sample_mask.copy()
    observed_mask[sample_mask] = noise_rng.random(measurements.shape) >= 0.08
    if not observed_mask.any():
        observed_mask[tuple(np.argwhere(sample_mask)[0])] = True
    sparse[~observed_mask] = 0
    return sparse, sample_mask, observed_mask


def _training_features(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rgb, target = _scene(seed, 64)
    scale = (2, 4, 8)[seed % 3]
    sparse, _, _ = _sample_sparse_depth(target, seed, scale)
    filled, original_mask, cleaned_mask = _denoise_and_fill(sparse)
    valid = cleaned_mask > 0
    low, high = float(np.min(sparse[valid])), float(np.max(sparse[valid]))
    unit_target = _to_unit(target, low, high)
    normalized_filled = _to_unit(filled, low, high)
    base = np.clip(cv2.resize(normalized_filled, (64, 64), interpolation=cv2.INTER_CUBIC), 0, 1)
    mask = cv2.resize(original_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
    rgb_unit = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
    inputs = np.concatenate((rgb_unit, base[None], mask[None]), axis=0).astype(np.float32)
    return inputs, (unit_target - base)[None].astype(np.float32)


@lru_cache(maxsize=1)
def _model() -> ResidualUNet:
    with _MODEL_LOCK:
        if not DEFAULT_CHECKPOINT.is_file():
            train_checkpoint(DEFAULT_CHECKPOINT)
        return load_checkpoint(DEFAULT_CHECKPOINT)


def _to_unit(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((depth - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def _edge_maps(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 40, 120) > 0
    depth_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(depth_x, depth_y)
    nonzero = magnitude[magnitude > 0.02]
    threshold = max(0.08, float(np.percentile(nonzero, 80))) if nonzero.size else 0.08
    depth_edges = magnitude >= threshold
    depth_edges = cv2.dilate(depth_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return rgb_edges, depth_edges


def _edge_aware_refine(
    rgb: np.ndarray,
    prediction: np.ndarray,
    initial_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_edges, depth_edges = _edge_maps(rgb, initial_depth)
    aligned_edges = rgb_edges & depth_edges
    protected = depth_edges | aligned_edges
    smooth = cv2.bilateralFilter(prediction.astype(np.float32), 5, 0.04, 2.0)
    blend = np.full(prediction.shape, 0.2, dtype=np.float32)
    blend[rgb_edges & ~protected] = 0.65
    blend[protected] = 0
    refined = prediction * (1 - blend) + smooth * blend
    edge_visualization = np.stack(
        (rgb_edges.astype(np.uint8) * 255, depth_edges.astype(np.uint8) * 255, aligned_edges.astype(np.uint8) * 255),
        axis=-1,
    )
    return refined.astype(np.float32), edge_visualization


def _edge_score(prediction: np.ndarray, guide: np.ndarray) -> float:
    depth_x = cv2.Sobel(prediction, cv2.CV_32F, 1, 0, ksize=3)
    depth_y = cv2.Sobel(prediction, cv2.CV_32F, 0, 1, ksize=3)
    guide_gray = _gray(guide)
    guide_x = cv2.Sobel(guide_gray, cv2.CV_32F, 1, 0, ksize=3)
    guide_y = cv2.Sobel(guide_gray, cv2.CV_32F, 0, 1, ksize=3)
    depth_magnitude = np.hypot(depth_x, depth_y)
    guide_magnitude = np.hypot(guide_x, guide_y)
    if depth_magnitude.std() < 1e-8 or guide_magnitude.std() < 1e-8:
        return 0.0
    correlation = np.corrcoef(depth_magnitude.ravel(), guide_magnitude.ravel())[0, 1]
    return float(np.clip((correlation + 1) / 2, 0, 1)) if np.isfinite(correlation) else 0.0


def _edge_f1(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_x = cv2.Sobel(prediction, cv2.CV_32F, 1, 0, ksize=3)
    pred_y = cv2.Sobel(prediction, cv2.CV_32F, 0, 1, ksize=3)
    truth_x = cv2.Sobel(ground_truth, cv2.CV_32F, 1, 0, ksize=3)
    truth_y = cv2.Sobel(ground_truth, cv2.CV_32F, 0, 1, ksize=3)
    dynamic_range = max(float(np.max(ground_truth) - np.min(ground_truth)), 1e-6)
    pred_edges = np.hypot(pred_x, pred_y) > dynamic_range * 0.08
    truth_edges = np.hypot(truth_x, truth_y) > dynamic_range * 0.08
    if not pred_edges.any() or not truth_edges.any():
        return 0.0
    tolerance = np.ones((3, 3), dtype=np.uint8)
    truth_neighborhood = cv2.dilate(truth_edges.astype(np.uint8), tolerance) > 0
    pred_neighborhood = cv2.dilate(pred_edges.astype(np.uint8), tolerance) > 0
    precision = float(np.mean(truth_neighborhood[pred_edges]))
    recall = float(np.mean(pred_neighborhood[truth_edges]))
    return 2 * precision * recall / max(precision + recall, 1e-8)


def calculate_metrics(prediction: np.ndarray, ground_truth: np.ndarray, guide: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(np.float32)
    ground_truth = ground_truth.astype(np.float32)
    residual = prediction - ground_truth
    dynamic_range = max(float(np.max(ground_truth) - np.min(ground_truth)), 1e-6)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    psnr = float(20 * np.log10(dynamic_range / max(rmse, 1e-8)))
    mean_p = cv2.GaussianBlur(prediction, (11, 11), 1.5)
    mean_g = cv2.GaussianBlur(ground_truth, (11, 11), 1.5)
    variance_p = cv2.GaussianBlur(prediction * prediction, (11, 11), 1.5) - mean_p * mean_p
    variance_g = cv2.GaussianBlur(ground_truth * ground_truth, (11, 11), 1.5) - mean_g * mean_g
    covariance = cv2.GaussianBlur(prediction * ground_truth, (11, 11), 1.5) - mean_p * mean_g
    c1, c2 = (0.01 * dynamic_range) ** 2, (0.03 * dynamic_range) ** 2
    ssim_map = ((2 * mean_p * mean_g + c1) * (2 * covariance + c2)) / (
        (mean_p * mean_p + mean_g * mean_g + c1) * (variance_p + variance_g + c2)
    )
    truth_x = cv2.Sobel(ground_truth, cv2.CV_32F, 1, 0, ksize=3)
    truth_y = cv2.Sobel(ground_truth, cv2.CV_32F, 0, 1, ksize=3)
    boundary = np.hypot(truth_x, truth_y) > dynamic_range * 0.08
    boundary_band = cv2.dilate(
        boundary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    if boundary_band.any():
        boundary_residual = residual[boundary_band]
        boundary_rmse = float(np.sqrt(np.mean(boundary_residual * boundary_residual)))
        boundary_mae = float(np.mean(np.abs(boundary_residual)))
    else:
        boundary_rmse = 0.0
        boundary_mae = 0.0
    if boundary.any():
        pred_x = cv2.Sobel(prediction, cv2.CV_32F, 1, 0, ksize=3)
        pred_y = cv2.Sobel(prediction, cv2.CV_32F, 0, 1, ksize=3)
        boundary_accuracy = float(np.mean(np.abs(np.hypot(pred_x, pred_y)[boundary] - np.hypot(truth_x, truth_y)[boundary]) < dynamic_range * 0.12))
    else:
        boundary_accuracy = 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "ssim": float(np.clip(np.mean(ssim_map), -1, 1)),
        "edge_preservation": _edge_score(prediction, guide),
        "boundary_accuracy": boundary_accuracy,
        "boundary_rmse": boundary_rmse,
        "boundary_mae": boundary_mae,
        "edge_f1": _edge_f1(prediction, ground_truth),
    }


def reconstruct(
    rgb: np.ndarray,
    sparse_depth: np.ndarray,
    method: str,
    scale: int = 4,
    radius: int = 5,
    epsilon: float = 0.0001,
    variant: dict[str, bool] | None = None,
) -> tuple[np.ndarray, float]:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB guidance must have three channels.")
    if sparse_depth.ndim != 2 or sparse_depth.size == 0:
        raise ValueError("Depth input must be a non-empty single-channel image.")
    if scale not in (2, 4, 8):
        raise ValueError("Upsampling scale must be 2, 4, or 8.")

    start = time.perf_counter()
    filled, original_valid_mask, cleaned_valid_mask = _denoise_and_fill(sparse_depth)
    valid_values = filled[cleaned_valid_mask > 0]
    low, high = float(np.min(valid_values)), float(np.max(valid_values))
    if high - low < 1e-6:
        high = low + 1.0
    normalized = _to_unit(filled, low, high)
    output_size = (
        min(rgb.shape[1], max(1, sparse_depth.shape[1] * scale)),
        min(rgb.shape[0], max(1, sparse_depth.shape[0] * scale)),
    )
    if output_size != (rgb.shape[1], rgb.shape[0]):
        rgb = cv2.resize(rgb, output_size, interpolation=cv2.INTER_AREA)
    target_size = (rgb.shape[1], rgb.shape[0])
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    initial_depth = np.clip(cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC), 0, 1)
    initial_depth = np.clip(initial_depth, 0, 1)
    valid_mask = cv2.resize(original_valid_mask, target_size, interpolation=cv2.INTER_NEAREST)
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }

    if method in interpolation:
        prediction = cv2.resize(normalized, target_size, interpolation=interpolation[method])
    else:
        if method == "joint_bilateral":
            prediction = _joint_bilateral(rgb, initial_depth)
        elif method == "guided":
            prediction = _guided_filter(_gray(rgb), initial_depth, radius, epsilon)
        else:
            base = initial_depth
            if method == "hybrid":
                options = variant or {}
                if options.get("use_guided", True):
                    base = _guided_filter(_gray(rgb), initial_depth, radius, epsilon)
                prediction = base
                if options.get("use_unet", True):
                    residual = _model().predict(rgb, base, valid_mask)
                    prediction = np.clip(base + residual, 0, 1)
                if options.get("use_refine", True):
                    prediction, _ = _edge_aware_refine(rgb, prediction, base)
            else:
                residual = _model().predict(rgb, base, valid_mask)
                prediction = np.clip(base + residual, 0, 1)

    prediction = np.clip(prediction, 0, 1) * (high - low) + low
    elapsed_ms = (time.perf_counter() - start) * 1000
    return prediction.astype(np.float32), elapsed_ms


def process_comparison(
    rgb: np.ndarray,
    sparse_depth: np.ndarray,
    ground_truth: np.ndarray | None = None,
    scale: int = 4,
    radius: int = 5,
    epsilon: float = 0.0001,
    methods: tuple[str, ...] | None = None,
) -> list[dict]:
    outputs = []
    target_size = (
        min(rgb.shape[1], sparse_depth.shape[1] * scale),
        min(rgb.shape[0], sparse_depth.shape[0] * scale),
    )
    guide = rgb if target_size == (rgb.shape[1], rgb.shape[0]) else cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    for method in methods or METHODS:
        prediction, latency = reconstruct(rgb, sparse_depth, method, scale, radius, epsilon)
        rgb_edges, depth_edges = _edge_maps(guide, prediction)
        edge_visualization = np.stack(
            (
                rgb_edges.astype(np.uint8) * 255,
                depth_edges.astype(np.uint8) * 255,
                np.logical_and(rgb_edges, depth_edges).astype(np.uint8) * 255,
            ),
            axis=-1,
        )
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "prediction": prediction,
            "latency_ms": latency,
            "edge_preservation": _edge_score(prediction, guide),
            "edge_visualization": edge_visualization,
            "process_rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        }
        if ground_truth is not None:
            resized_truth = cv2.resize(
                ground_truth.astype(np.float32),
                (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            row.update(calculate_metrics(prediction, resized_truth, rgb))
        outputs.append(row)
    return outputs
