# Design document — DepthLab

## 1. Goal and workflow

DepthLab reconstructs dense depth from sparse or low-resolution measurements, using an RGB image as guidance without treating RGB texture as a depth-value source. A user uploads RGB and depth images, optionally supplies dense ground truth, chooses a method and upsampling scale, inspects results and metrics, and downloads a 16-bit depth PNG.

The dashboard also reports measured synthetic evaluation data. Without a supplied dense reference, supervised accuracy metrics are unavailable rather than fabricated. Uploads are processed in memory and not persisted.

## 2. Requirement mapping

| Requirement | Implementation | Evidence |
|---|---|---|
| Classical methods | Nearest, bilinear, bicubic, joint bilateral, guided filter | `app/depth.py` |
| Learned refinement | Compact residual encoder-decoder U-Net with skip connections | `app/model.py` |
| Hybrid refinement | Bicubic → guided filter → U-Net residual → edge-aware refinement | `app/depth.py` |
| Missing/noisy depth | Finite/positive validity mask, outlier removal, hole filling, synthetic noise/dropout | `app/depth.py` |
| 2×, 4×, 8× support | Output sizing and scale-aware benchmark | `app/depth.py`, `app/benchmark.py` |
| Evaluation | Accuracy including twice-dilated boundary RMSE/MAE, edge metrics, timing/RSS, std, robustness, seven-row ablation/reference | `app/benchmark.py`, `evaluation/results.json` |
| Demonstration | FastAPI API and responsive HTML/CSS/JavaScript client | `app/main.py`, `static/` |
| Reproducibility | Seeded model training, stored weights, tests, Dockerfile | `app/model.py`, `models/`, `tests/`, `Dockerfile` |

## 3. Architecture

```text
Browser dashboard (HTML / CSS / JavaScript)
  ├─ sample, multipart uploads, method + scale controls
  ├─ results, edge/error visualization, metric tables, PNG downloads
  └─ GET /api/health | /api/demo | /api/benchmark
                    │ same-origin HTTP
                    ▼
FastAPI service (app/main.py)
  ├─ input validation and in-memory image decoding
  ├─ POST /api/process and PNG/JSON response
  ▼
Depth pipeline (app/depth.py)
  ├─ preserve validity mask, remove isolated outliers, fill holes
  ├─ five classical/edge-aware baselines
  ├─ Residual U-Net and bicubic → guided → U-Net → edge-aware hybrid
  └─ accuracy/edge metrics, latency, process RSS
                    │
                    ▼
Held-out benchmark (app/benchmark.py) ── evaluation/results.json
  ├─ 7 methods × 3 scales, mean/std
  ├─ noisy and missing-depth experiments
  └─ hybrid ablation and measured failure notes
```

See [ARCHITECTURE.svg](./ARCHITECTURE.svg). The app has no database; recent-run summaries are stored only in the current browser.

## 4. Model and preprocessing

The Residual U-Net receives five channels: normalized RGB, an initial upsampled depth, and the original measurement-validity mask. Its encoder and decoder use residual blocks and skip connections. It predicts a depth correction, which is added to the initial depth. Model inputs are built without dense ground truth.

Training examples use controlled phase-shifted sparse grids, depth-dependent Gaussian measurement noise, and 8% dropouts among sampled pixels. Unmeasured pixels remain invalid before preprocessing. Dense targets are kept separately for the residual loss. Training uses seeds 100–119; validation uses seeds 120–123 for checkpoint selection and early stopping. The model weights are saved atomically to `models/residual_unet.pt`, loaded once per process, and cached across API requests. Explicit retraining is available through `python -m app.train_model`.

Preprocessing marks non-finite and non-positive values invalid, preserves the original validity mask, removes isolated measurement spikes, and fills missing values from nearest valid samples for interpolation. If there are no valid measurements, processing returns an explicit validation error.

## 5. Reconstruction methods

- **Nearest, bilinear, bicubic:** interpolate normalized, hole-filled depth at the requested scale.
- **Joint bilateral:** combine spatial, RGB-color, and depth-range weights.
- **Guided filter:** apply a local linear guide-to-depth model with configurable radius and regularisation epsilon (default 0.0001).
- **Residual U-Net:** predict a correction from RGB, bicubic initial depth, and validity mask.
- **Hybrid:** initialize with bicubic depth, apply guided filtering, predict and add a U-Net residual, then perform edge-aware refinement. The standalone U-Net result remains separately selectable.

RGB edges and depth gradients are visualized together; edge-aware refinement is constrained by both the guide and depth structure to reduce texture copying. Edge metrics are agreement proxies, not proof of correct geometry.

## 6. Evaluation protocol

The evaluation is synthetic because no real dataset was supplied. It uses eight deterministic held-out 128×128 scenes per scale, with disjoint training/validation/evaluation seeds 100–119, 120–123, and 2100–2107. All seven methods are compared at 2×, 4×, and 8×.

Metrics are RMSE, MAE, PSNR, local-window SSIM, RGB/depth edge-preservation correlation, boundary accuracy, and edge F1. The benchmark reports mean and population standard deviation across scenes. Boundary RMSE/MAE are computed over the ground-truth Sobel edge mask dilated twice with a 3×3 kernel. It also measures wall-clock processing latency, actual process-wide RSS through psutil, and Python-traced allocation (which excludes some native allocations).

Additional experiments use 36 mm measurement noise and 20% missing input at 4×. The seven 4× ablation/reference rows are full hybrid, without guided filtering, without U-Net, without edge refinement, without RGB (constant gray guide), bicubic reference, and joint bilateral reference. Results are descriptive and not real-world claims. See [evaluation/results.json](../evaluation/results.json) and [eval_P4_T54.md](../eval_P4_T54.md).

## 7. API and validation

`GET /api/health`, `/api/demo`, and `/api/benchmark` supply service status, generated sample data, and cached evaluation results. `POST /api/process` accepts optional multipart RGB guidance, required depth, optional ground truth, method, scale, radius, epsilon, and a simulation flag. In simulation mode, the uploaded depth is treated as dense ground truth, cropped to a scale multiple, and downsampled with 1% Gaussian noise and 5% missing samples. When RGB is omitted, a normalized grayscale guide is derived from the depth measurements.

Images are decoded and processed in memory. Public deployments should add authentication, request throttling, and operational monitoring before accepting untrusted traffic.

## 8. Testing and known limitations

`python -m pytest -q` covers sparse-data generation/leakage, preprocessing, model training and checkpoint loading, all methods and scales, and API error/success paths. `python -m app.benchmark --output evaluation/results.json` regenerates the report.

The model is trained only on simple generated geometry. Realistic sensor calibration, uncertainty, indoor/outdoor variation, reflective surfaces, thin structures, and RGB/depth misregistration are untested. On the current synthetic split, guided filtering has slightly lower RMSE, PSNR, and boundary RMSE than the hybrid at 4× and 8×, while the hybrid has slightly higher SSIM. Joint bilateral leads 4× MAE and edge F1 and 2× SSIM. Removing guided filtering returns hybrid RMSE to bicubic level, while removing the U-Net changes hybrid 4× RMSE by less than 0.3 mm. These synthetic results are not comparable to NYU or KITTI.
