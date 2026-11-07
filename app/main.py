from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .benchmark import run_benchmark
from .data import demo_scene
from .depth import DEFAULT_CHECKPOINT, METHODS, METHOD_LABELS, _filled_depth, process_comparison


ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 1024 * 1024

app = FastAPI(title="DepthLab", version="1.0.0", description="Edge-aware depth map reconstruction and evaluation.")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def _png_b64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode a result image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _colorize(depth: np.ndarray) -> np.ndarray:
    low, high = float(np.min(depth)), float(np.max(depth))
    normalized = np.clip((depth - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _decode_rgb(payload: bytes, field: str) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail=f"{field} is not a supported image.")
    if decoded.shape[0] * decoded.shape[1] > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"{field} exceeds the 1-megapixel processing limit.")
    if min(decoded.shape[:2]) < 2:
        raise HTTPException(status_code=400, detail=f"{field} must be at least 2 × 2 pixels.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _decode_depth(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Depth input is not a supported image.")
    if decoded.shape[0] * decoded.shape[1] > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Depth input exceeds the 1-megapixel processing limit.")
    if decoded.ndim == 3:
        if decoded.shape[2] == 4:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2GRAY)
        elif decoded.shape[2] == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        else:
            raise HTTPException(status_code=400, detail="Depth input must be a single-channel image.")
    depth = decoded.astype(np.float32)
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise HTTPException(status_code=400, detail="Depth input must contain positive measurements; zero is treated as missing.")
    return depth


def _depth_guide(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise HTTPException(status_code=400, detail="Depth input contains no valid measurements for guidance.")
    filled, _ = _filled_depth(depth)
    low, high = float(np.min(depth[valid])), float(np.max(depth[valid]))
    if high - low <= 1e-6:
        normalized = np.zeros_like(filled, dtype=np.float32)
    else:
        normalized = np.clip((filled - low) / (high - low), 0, 1)
    gray = cv2.resize(
        (normalized * 255).astype(np.uint8),
        size,
        interpolation=cv2.INTER_CUBIC,
    )
    return np.repeat(gray[:, :, None], 3, axis=2)


def _simulated_sparse_depth(ground_truth: np.ndarray, scale: int) -> np.ndarray:
    height, width = ground_truth.shape
    sparse_height, sparse_width = height // scale, width // scale
    sparse = cv2.resize(
        ground_truth,
        (sparse_width, sparse_height),
        interpolation=cv2.INTER_AREA,
    )
    rng = np.random.default_rng(2026)
    depth_range = float(np.max(ground_truth) - np.min(ground_truth))
    noise_sigma = max(depth_range * 0.01, np.finfo(np.float32).eps)
    sparse = np.maximum(
        sparse + rng.normal(0, noise_sigma, sparse.shape).astype(np.float32),
        np.finfo(np.float32).eps,
    )
    sparse[rng.random(sparse.shape) < 0.05] = 0
    return sparse.astype(np.float32)


async def _read_image(file: UploadFile, field: str) -> bytes:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail=f"{field} is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{field} exceeds the 8 MB upload limit.")
    return payload


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "methods": list(METHODS),
        "max_image_pixels": MAX_PIXELS,
        "model": "Residual U-Net",
        "model_checkpoint_available": DEFAULT_CHECKPOINT.is_file(),
    }


@app.get("/api/demo")
def demo() -> dict:
    rgb, sparse, truth = demo_scene()
    raw_sparse = np.clip(sparse, 0, 65535).astype(np.uint16)
    raw_truth = np.clip(truth, 0, 65535).astype(np.uint16)
    return {
        "rgb": _png_b64(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
        "sparse_depth": _png_b64(raw_sparse),
        "ground_truth": _png_b64(raw_truth),
        "shape": [int(rgb.shape[1]), int(rgb.shape[0])],
        "scale": 4,
        "depth_unit": "mm",
        "description": "Generated indoor scene with sparse, noisy depth and known ground truth.",
    }


@app.get("/api/benchmark")
async def benchmark() -> dict:
    return await run_in_threadpool(run_benchmark)


@app.post("/api/process")
async def process(
    depth_file: Annotated[UploadFile, File()],
    rgb_file: Annotated[UploadFile | None, File()] = None,
    ground_truth_file: Annotated[UploadFile | None, File()] = None,
    method: Annotated[str, Form()] = "hybrid",
    scale: Annotated[int, Form()] = 4,
    radius: Annotated[int, Form()] = 5,
    epsilon: Annotated[float, Form()] = 0.0001,
    simulate: Annotated[bool, Form()] = False,
) -> dict:
    if method != "compare" and method not in METHODS:
        raise HTTPException(status_code=400, detail="Choose a supported method or compare all methods.")
    if scale not in (2, 4, 8):
        raise HTTPException(status_code=400, detail="Upsampling scale must be 2, 4, or 8.")
    if not 1 <= radius <= 12 or not 0.00001 <= epsilon <= 0.1:
        raise HTTPException(status_code=400, detail="Filter radius or epsilon is outside the supported range.")

    uploaded_depth = _decode_depth(await _read_image(depth_file, "Depth image"))
    if uploaded_depth.shape[0] < 2 or uploaded_depth.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Depth input must be at least 2 × 2 pixels.")
    uploaded_rgb = (
        _decode_rgb(await _read_image(rgb_file, "RGB image"), "RGB image")
        if rgb_file is not None and rgb_file.filename
        else None
    )

    if simulate:
        if not np.isfinite(uploaded_depth).all() or np.any(uploaded_depth <= 0):
            raise HTTPException(
                status_code=400,
                detail="Simulation requires a dense depth image with finite, positive ground-truth values.",
            )
        crop_height = uploaded_depth.shape[0] // scale * scale
        crop_width = uploaded_depth.shape[1] // scale * scale
        if crop_height < 2 * scale or crop_width < 2 * scale:
            raise HTTPException(
                status_code=400,
                detail="Depth image must be at least twice the selected scale in both dimensions for simulation.",
            )
        truth = uploaded_depth[:crop_height, :crop_width].copy()
        if uploaded_rgb is not None:
            rgb_ratio = uploaded_rgb.shape[1] / uploaded_rgb.shape[0]
            depth_ratio = truth.shape[1] / truth.shape[0]
            if abs(rgb_ratio - depth_ratio) / max(rgb_ratio, depth_ratio) > 0.05:
                raise HTTPException(status_code=400, detail="RGB and depth images must have matching aspect ratios.")
            rgb = cv2.resize(uploaded_rgb, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
        else:
            rgb = None
        sparse_depth = _simulated_sparse_depth(truth, scale)
        if rgb is None:
            rgb = _depth_guide(sparse_depth, (crop_width, crop_height))
    else:
        sparse_depth = uploaded_depth
        truth = None
        if ground_truth_file is not None and ground_truth_file.filename:
            truth = _decode_depth(await _read_image(ground_truth_file, "Ground-truth image"))
        if uploaded_rgb is not None:
            rgb_ratio = uploaded_rgb.shape[1] / uploaded_rgb.shape[0]
            depth_ratio = sparse_depth.shape[1] / sparse_depth.shape[0]
            if abs(rgb_ratio - depth_ratio) / max(rgb_ratio, depth_ratio) > 0.05:
                raise HTTPException(status_code=400, detail="RGB and depth images must have matching aspect ratios.")
            rgb = uploaded_rgb
        else:
            output_width = min(sparse_depth.shape[1] * scale, MAX_PIXELS)
            output_height = min(sparse_depth.shape[0] * scale, MAX_PIXELS)
            if output_width * output_height > MAX_PIXELS:
                ratio = (MAX_PIXELS / (output_width * output_height)) ** 0.5
                output_width = max(2, int(output_width * ratio))
                output_height = max(2, int(output_height * ratio))
            rgb = _depth_guide(sparse_depth, (output_width, output_height))
        if truth is not None:
            rgb_ratio = rgb.shape[1] / rgb.shape[0]
            truth_ratio = truth.shape[1] / truth.shape[0]
            if abs(rgb_ratio - truth_ratio) / max(rgb_ratio, truth_ratio) > 0.05:
                raise HTTPException(status_code=400, detail="Ground-truth and RGB images must have matching aspect ratios.")
    selected_methods = METHODS if method == "compare" else (method,)

    try:
        rows = await run_in_threadpool(
            process_comparison,
            rgb,
            sparse_depth,
            truth,
            scale,
            radius,
            epsilon,
            selected_methods,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=f"Residual U-Net processing failed: {error}") from error

    results = []
    for row in rows:
        if row["method"] not in selected_methods:
            continue
        depth = row["prediction"]
        metrics = {
            key: row[key]
            for key in (
                "rmse",
                "mae",
                "psnr",
                "ssim",
                "edge_preservation",
                "boundary_accuracy",
                "boundary_rmse",
                "boundary_mae",
                "edge_f1",
            )
            if key in row
        }
        results.append(
            {
                "method": row["method"],
                "label": row["label"],
                "latency_ms": row["latency_ms"],
                "process_rss_mb": row["process_rss_mb"],
                "metrics": metrics,
                "metrics_available": truth is not None,
                "depth_png": _png_b64(np.clip(depth, 0, 65535).astype(np.uint16)),
                "color_png": _png_b64(_colorize(depth)),
                "edge_png": _png_b64(row["edge_visualization"]),
                "error_png": _png_b64(_colorize(np.abs(depth - cv2.resize(truth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)))) if truth is not None else None,
            }
        )

    return {
        "results": results,
        "input": {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "depth_width": int(sparse_depth.shape[1]),
            "depth_height": int(sparse_depth.shape[0]),
            "scale": scale,
            "has_ground_truth": truth is not None,
            "simulated": simulate,
        },
        "unit": "input units",
        "methods": [METHOD_LABELS[key] for key in selected_methods],
    }
