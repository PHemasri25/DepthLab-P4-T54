---
title: DepthLab
emoji: 🗺️
sdk: docker
app_port: 8000
---
# DepthLab — Edge-preserving depth map reconstruction

DepthLab is a local-first computer-vision workbench for refining sparse or low-resolution depth maps with RGB guidance. It provides five classical baselines, a trained Residual U-Net, and a hybrid pipeline, plus an interactive FastAPI dashboard and a reproducible synthetic benchmark.

> **Evaluation scope:** The benchmark is deterministic and synthetic. No real sensor dataset or hosting credentials were supplied. Results must not be interpreted as real-world accuracy. Team T54 USNs are listed in [eval_P4_T54.md](eval_P4_T54.md).

## Features

- Upload RGB guidance, sparse depth, and optional dense ground truth, or load a generated sample.
- Select nearest, bilinear, bicubic, joint bilateral, guided filter, Residual U-Net, or hybrid; run a side-by-side comparison.
- Select 2×, 4×, or 8× output scale and adjust guided-filter radius.
- Preserve the original validity mask, reject invalid inputs, fill holes, and remove isolated depth spikes.
- Inspect colorized depth, RGB/depth edge overlays, optional error maps, and downloadable 16-bit depth PNGs.
- View measured RMSE, MAE, PSNR, SSIM, edge preservation, boundary accuracy, edge F-score, processing latency, and process RSS when ground truth is available.
- Review held-out results at each scale, mean ± standard deviation, noisy/missing-depth experiments, an ablation, and failure analysis.
- Uploads are processed in memory and are not persisted by the API.

## Requirements and local launch

Python 3.10+ and pip are required. On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000); interactive API docs are at `/docs`.

The trained checkpoint is included at `models/residual_unet.pt` and is loaded once per process; API requests do not retrain the model. To explicitly retrain and overwrite it:

```powershell
python -m app.train_model
```

The training pipeline uses deterministic, controlled sparse sampling, noisy valid measurements, and dropouts. The dense depth is used only as the separate residual target and to simulate sparse measurements; it is not included in model input features. Validation scenes are separate from training scenes and select the checkpoint with early stopping.

The server binds to localhost by default. Use `0.0.0.0` only when intentionally exposing it on a trusted network.

## Docker

```powershell
docker build -t depthlab .
docker run --rm -p 8000:8000 depthlab
```

The image includes CPU PyTorch and the trained checkpoint. The container listens on port 8000.

## Problem statement and rubric map

| Requirement | Implementation |
|---|---|
| Classical upsampling | Nearest, bilinear, bicubic, joint bilateral, guided filter |
| Learned refinement | Compact residual encoder-decoder U-Net with skip connections; RGB, initial depth, and original validity mask inputs |
| Hybrid reconstruction | Bicubic initialization → guided filtering → U-Net residual → edge-aware refinement |
| Sparse/noisy/missing depth | Controlled synthetic sampling, invalid-value handling, preserved validity mask, hole filling and isolated-spike removal |
| Multi-scale evaluation | All seven methods at 2×, 4×, and 8× |
| Quantitative evaluation | RMSE, MAE, PSNR, SSIM, edge preservation, boundary accuracy, edge F1, latency, process RSS, and Python allocation |
| Robustness and analysis | High-noise and high-missing experiments, mean/std aggregation, component ablation, generated failure analysis |
| Functional demo | FastAPI upload API and HTML/CSS/JavaScript dashboard with visualizations, loading/error states, comparison, and downloadable outputs |
| Reproducibility | Checkpointed model, tests, Dockerfile, design/architecture docs, and generated JSON report |

No ground-truth accuracy metrics are fabricated for uploads without a reference map. Process RSS is the actual process-wide resident set size, not memory uniquely attributable to a method; Python allocation is reported separately.

## API

- `GET /api/health` — service status, supported methods, and checkpoint availability.
- `GET /api/demo` — generated RGB, sparse depth, and reference PNGs.
- `GET /api/benchmark` — cached benchmark measurements for the current process.
- `POST /api/process` — multipart `rgb_file`, `depth_file`, optional `ground_truth_file`, plus `method`, `scale`, `radius`, and `epsilon`. Use `method=compare` for all methods.

Images are capped at 8 MB and one megapixel each. Zero and non-finite depth samples are invalid; at least one positive measurement is required. Input units are preserved in output; the generated scene uses millimetres. Downloads use unsigned 16-bit PNG.

## Evaluation and tests

```powershell
python -m app.benchmark --output evaluation/results.json
python -m pytest -q
```

The benchmark uses eight held-out 128×128 synthetic scenes per scale, with training seeds 100–119, validation seeds 120–123, and evaluation seeds 2100–2107. Noise and missing-value experiments, ablation, and failure notes are generated from the same held-out split. Runtime and memory vary by machine. Python-traced allocation does not include every native OpenCV allocation.

The actual generated measurements are in [evaluation/results.json](evaluation/results.json) and summarized in [eval_P4_T54.md](eval_P4_T54.md). No real RGB-D dataset is included; the report explicitly labels results synthetic.

## Limitations

- The learned model is trained only on simple generated scenes, not NYU Depth, KITTI, or another real sensor dataset.
- Generated geometry does not capture realistic sensor noise, calibration, reflective/transparent surfaces, or RGB/depth misregistration.
- RGB texture and shadows can be mistaken for depth boundaries; edge agreement is not a substitute for ground-truth accuracy.
- The current synthetic evaluation shows the classical joint bilateral method outperforms the hybrid on RMSE. The hybrid is retained for comparison, not presented as the best method.
- Validate on authorized, scene-disjoint real data before using results in consequential applications. Depth maps can expose sensitive spatial information.

## Submission and deployment

The workspace has no configured Git remote or public hosting credentials. Add the team repository and deployed application URLs to the evaluation submission after publishing through an approved host.
