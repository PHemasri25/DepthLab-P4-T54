from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
import psutil

from .data import benchmark_scene
from .depth import METHODS, METHOD_LABELS, _model, calculate_metrics, reconstruct


@lru_cache(maxsize=1)
def run_benchmark() -> dict:
    _model()
    scales = (2, 4, 8)
    results_by_scale: dict[str, list[dict]] = {}
    ablation_rows: dict[str, list[dict[str, float]]] = {}
    experiment_rows: dict[str, dict[str, list[dict[str, float]]]] = {
        "high_noise": {method: [] for method in METHODS},
        "high_missing": {method: [] for method in METHODS},
    }
    scene_count = 8

    for scale in scales:
        per_method: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
        peak_by_method: dict[str, list[float]] = {method: [] for method in METHODS}

        for seed in range(2100, 2100 + scene_count):
            rgb, sparse, truth = benchmark_scene(seed, scale=scale)
            for method in METHODS:
                tracemalloc.start()
                started = time.perf_counter()
                prediction, _ = reconstruct(rgb, sparse, method, scale)
                elapsed_ms = (time.perf_counter() - started) * 1000
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                metrics = calculate_metrics(prediction, truth, rgb)
                metrics["latency_ms"] = elapsed_ms
                metrics["process_rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
                per_method[method].append(metrics)
                peak_by_method[method].append(peak / (1024 * 1024))

            if scale == 4:
                if not ablation_rows:
                    ablation_rows = {
                        "Full hybrid (bicubic + guided + U-Net)": [],
                        "Without U-Net (joint bilateral only)": [],
                        "Without RGB guidance": [],
                    }
                for label, guide in (
                    ("Full hybrid (bicubic + guided + U-Net)", rgb),
                    ("Without U-Net (joint bilateral only)", rgb),
                    ("Without RGB guidance", np.full_like(rgb, 128)),
                ):
                    method = "hybrid" if label != "Without U-Net (joint bilateral only)" else "joint_bilateral"
                    started = time.perf_counter()
                    prediction, _ = reconstruct(guide, sparse, method, scale)
                    metrics = calculate_metrics(prediction, truth, rgb)
                    metrics["latency_ms"] = (time.perf_counter() - started) * 1000
                    metrics["process_rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
                    ablation_rows[label].append(metrics)

        rows_for_scale = []
        for method in METHODS:
            rows = per_method[method]
            rows_for_scale.append(_aggregate(method, rows, peak_by_method[method]))
        results_by_scale[str(scale)] = rows_for_scale

    for seed in range(2100, 2100 + scene_count):
        experiments = {
            "high_noise": benchmark_scene(seed, scale=4, noise_std_mm=36.0, missing_rate=0.04),
            "high_missing": benchmark_scene(seed, scale=4, noise_std_mm=18.0, missing_rate=0.2),
        }
        for experiment_name, (rgb, sparse, truth) in experiments.items():
            for method in METHODS:
                started = time.perf_counter()
                prediction, _ = reconstruct(rgb, sparse, method, 4)
                elapsed_ms = (time.perf_counter() - started) * 1000
                metrics = calculate_metrics(prediction, truth, rgb)
                metrics["latency_ms"] = elapsed_ms
                metrics["process_rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
                experiment_rows[experiment_name][method].append(metrics)

    experiment_results = {}
    for name, per_method in experiment_rows.items():
        experiment_results[name] = [
            _aggregate(method, rows)
            for method, rows in per_method.items()
        ]

    return {
        "dataset": "Procedurally generated, held-out indoor depth scenes",
        "scenes": scene_count,
        "scale": 4,
        "scales": list(scales),
        "unit": "mm",
        "split": "CNN training seeds 100–119; validation seeds 120–123; evaluation seeds 2100–2107 (disjoint)",
        "limitations": "Synthetic scenes are not a substitute for a real sensor dataset; latency and Python allocation measurements are machine-dependent.",
        "results": results_by_scale["4"],
        "results_by_scale": results_by_scale,
        "experiments": {
            "scenes": scene_count,
            "conditions": {
                "high_noise": {"noise_std_mm": 36.0, "missing_rate": 0.04},
                "high_missing": {"noise_std_mm": 18.0, "missing_rate": 0.2},
            },
            "results": experiment_results,
        },
        "ablation": [
            {"variant": label, **_aggregate(label, rows)}
            for label, rows in ablation_rows.items()
        ],
        "failure_analysis": _failure_analysis(results_by_scale, experiment_results),
    }


def _aggregate(
    method: str,
    rows: list[dict[str, float]],
    python_peak_values: list[float] | None = None,
) -> dict:
    metric_names = (
        "rmse",
        "mae",
        "psnr",
        "ssim",
        "edge_preservation",
        "boundary_accuracy",
        "edge_f1",
        "latency_ms",
        "process_rss_mb",
    )
    aggregated = {"method": method, "label": METHOD_LABELS.get(method, method), "scenes": len(rows)}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregated[name] = float(np.mean(values))
        aggregated[f"{name}_std"] = float(np.std(values))
    if python_peak_values is not None:
        aggregated["python_peak_mb"] = float(np.mean(python_peak_values))
        aggregated["python_peak_mb_std"] = float(np.std(python_peak_values))
    return aggregated


def _failure_analysis(
    results_by_scale: dict[str, list[dict]],
    experiments: dict[str, list[dict]],
) -> list[str]:
    largest_scale = max(
        (2, 4, 8),
        key=lambda scale: next(row["rmse"] for row in results_by_scale[str(scale)] if row["method"] == "hybrid"),
    )
    hard_noise = max(
        (row for row in experiments["high_noise"] if row["method"] == "hybrid"),
        key=lambda row: row["rmse"],
    )
    hard_missing = max(
        (row for row in experiments["high_missing"] if row["method"] == "hybrid"),
        key=lambda row: row["rmse"],
    )
    return [
        f"On this synthetic split, the hybrid's largest tested factor ({largest_scale}×) is its hardest scale; higher sparsity increases boundary ambiguity.",
        f"At 4× with 36 mm input noise, the hybrid mean RMSE is {hard_noise['rmse']:.1f} mm (±{hard_noise['rmse_std']:.1f}).",
        f"At 4× with 20% missing input, the hybrid mean RMSE is {hard_missing['rmse']:.1f} mm (±{hard_missing['rmse_std']:.1f}).",
        "These generated scenes do not measure performance on real RGB-D sensors; thin structures, reflective surfaces, misregistration, and real noise distributions remain untested.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate depth reconstruction on held-out synthetic scenes.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()
    report = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {report['scenes']} unseen scenes; saved {args.output}")
    for row in sorted(report["results"], key=lambda item: item["rmse"]):
        print(
            f"{row['label']:<24} RMSE {row['rmse']:7.1f} mm  "
            f"MAE {row['mae']:6.1f} mm  SSIM {row['ssim']:.3f}  "
            f"{row['latency_ms']:6.1f} ms"
        )


if __name__ == "__main__":
    main()
