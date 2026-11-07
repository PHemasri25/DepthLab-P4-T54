1RUA24CSE0294, 1RUA24CSE0292, 1RUA24CSE0264, 1RUA24CSE0321, 1RUA24CSE0411

# P4 comparative evaluation — T54

> Add the team GitHub repository and deployed application URLs before submission. This report contains synthetic results only; no real RGB-D dataset was supplied.

## Evaluation setup

- **Problem:** P4 — Edge-Preserving Depth Map Upsampling and Refinement
- **Data:** deterministic procedurally generated RGB/depth scenes
- **Split:** training seeds 100–119; validation seeds 120–123; held-out evaluation seeds 2100–2107
- **Scenes:** eight 128×128 synthetic scenes at each scale
- **Scales:** 2×, 4×, and 8×
- **Depth units:** millimetres
- **Reproduce:** `python -m app.benchmark --output evaluation/results.json`
- **Aggregate:** arithmetic mean ± population standard deviation over scenes
- **Memory:** process RSS is actual process-wide resident set size, not memory uniquely attributable to each method. Python-traced allocation is reported separately and omits some native OpenCV allocations.
- **Runtime:** CPU measurements are machine-dependent and include reconstruction, not one-time model training.

## 4× comparison

Every number is mean ± standard deviation. Latency is milliseconds; memory is MB.

| Method | RMSE (mm) ↓ | MAE (mm) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge preservation ↑ | Boundary accuracy ↑ | Edge F1 ↑ | Latency (ms) | Process RSS (MB) | Python peak (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 212.66 ± 47.29 | 61.72 ± 12.17 | 20.10 ± 1.50 | 0.823 ± 0.016 | 68.1 ± 4.4% | 10.2 ± 3.1% | 51.8 ± 8.6% | 1.54 ± 0.57 | 258.69 ± 0.01 | 0.33 ± 0.00 |
| Bilinear | 183.15 ± 39.82 | 64.80 ± 14.00 | 21.38 ± 1.45 | 0.863 ± 0.016 | 73.9 ± 5.2% | 7.1 ± 4.1% | 66.8 ± 2.8% | 1.30 ± 0.36 | 258.69 ± 0.01 | 0.33 ± 0.00 |
| Bicubic | 180.07 ± 39.63 | 65.71 ± 13.85 | 21.53 ± 1.47 | 0.848 ± 0.017 | 74.8 ± 5.4% | 6.3 ± 3.5% | 58.4 ± 4.3% | 1.30 ± 0.46 | 258.69 ± 0.01 | 0.33 ± 0.00 |
| Joint bilateral | **158.51 ± 37.27** | **52.32 ± 11.67** | **22.68 ± 1.64** | **0.902 ± 0.015** | 80.2 ± 6.5% | 8.2 ± 4.8% | **68.5 ± 4.1%** | 36.99 ± 14.91 | 258.69 ± 0.01 | 1.36 ± 0.00 |
| Guided filter | 200.20 ± 50.73 | 95.52 ± 30.25 | 20.65 ± 1.64 | 0.834 ± 0.050 | **86.0 ± 7.0%** | 1.2 ± 2.0% | 47.5 ± 3.3% | 1.63 ± 0.53 | 258.69 ± 0.01 | 0.96 ± 0.00 |
| Residual U-Net | 180.19 ± 39.66 | 65.91 ± 13.88 | 21.53 ± 1.47 | 0.848 ± 0.017 | 74.8 ± 5.4% | 6.3 ± 3.5% | 58.4 ± 4.3% | 19.78 ± 5.54 | 258.69 ± 0.01 | 1.10 ± 0.00 |
| Hybrid (bicubic → guided → U-Net → edge refinement) | 200.35 ± 50.68 | 95.95 ± 30.24 | 20.64 ± 1.63 | 0.834 ± 0.050 | 86.0 ± 7.0% | 1.2 ± 2.0% | 47.5 ± 3.3% | 23.05 ± 5.28 | 258.69 ± 0.01 | 1.16 ± 0.00 |

## RMSE and SSIM across scales

| Method | 2× RMSE | 2× SSIM | 4× RMSE | 4× SSIM | 8× RMSE | 8× SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 135.13 | 0.888 | 212.66 | 0.823 | 302.30 | 0.738 |
| Bilinear | 120.45 | 0.926 | 183.15 | 0.863 | 263.00 | 0.773 |
| Bicubic | 116.53 | 0.911 | 180.07 | 0.848 | 259.30 | 0.757 |
| Joint bilateral | **92.53** | **0.969** | **158.51** | **0.902** | **243.38** | **0.794** |
| Guided filter | 180.32 | 0.855 | 200.20 | 0.834 | 251.04 | 0.786 |
| Residual U-Net | 116.66 | 0.911 | 180.19 | 0.848 | 259.35 | 0.757 |
| Hybrid | 180.44 | 0.854 | 200.35 | 0.834 | 251.14 | 0.786 |

RMSE is in millimetres. The complete per-scale results and standard deviations for every metric are recorded in `evaluation/results.json`.

## Noisy and missing-depth experiments

Mean RMSE ± standard deviation at 4× across eight synthetic scenes:

| Method | High noise: 36 mm (mm) | 20% missing (mm) |
|---|---:|---:|
| Nearest neighbour | 211.04 ± 41.14 | 238.43 ± 56.99 |
| Bilinear | 181.98 ± 34.80 | 203.97 ± 47.00 |
| Bicubic | 178.82 ± 33.08 | 206.85 ± 49.44 |
| Joint bilateral | **155.24 ± 29.84** | **187.77 ± 48.15** |
| Guided filter | 199.83 ± 48.83 | 210.26 ± 51.78 |
| Residual U-Net | 178.94 ± 33.14 | 206.92 ± 49.42 |
| Hybrid | 199.98 ± 48.80 | 210.40 ± 51.76 |

## Ablation study

4× synthetic held-out scenes; mean ± standard deviation where shown:

| Variant | RMSE (mm) ↓ | MAE (mm) ↓ | SSIM ↑ | Edge F1 ↑ |
|---|---:|---:|---:|---:|
| Full hybrid (bicubic + guided + U-Net) | 200.35 ± 50.68 | 95.95 ± 30.24 | 0.834 ± 0.050 | 47.5 ± 3.3% |
| Without U-Net (joint bilateral only) | **158.51 ± 37.27** | **52.32 ± 11.67** | **0.902 ± 0.015** | **68.5 ± 4.1%** |
| Without RGB guidance (constant gray guide) | 242.15 ± 50.96 | 117.70 ± 28.62 | 0.777 ± 0.031 | 44.3 ± 3.4% |

## Interpretation and failure analysis

- Joint bilateral filtering is the best RMSE method in these synthetic measurements. The hybrid did not improve on its classical joint bilateral baseline and performs materially worse at 4×. It remains implemented as the requested staged pipeline and is exposed for direct comparison, not claimed as the top method.
- The standalone Residual U-Net is close to bicubic on this generated split. Synthetic training and simple geometry are insufficient to establish real-scene generalization.
- The hybrid's highest RMSE among tested factors is at 8×. At 4× its RMSE is 200.0 ± 48.8 mm with 36 mm input noise and 210.4 ± 51.8 mm with 20% missing input.
- Guided filtering has high RGB/depth edge-preservation correlation but low boundary accuracy; those are different metrics and neither alone establishes accurate geometry.
- Simple generated scenes do not measure sensor calibration, realistic indoor geometry, thin structures, reflective/transparent surfaces, or RGB/depth misregistration.
- Nearest-valid hole filling can propagate an incorrect measurement through a large missing region. The system does not estimate uncertainty.
- No real-dataset results are available; do not compare these numbers to published NYU/KITTI results.

## Demo checklist

- [ ] Public application URL works on a first click.
- [ ] Run the generated sample and at least one authorized unseen image.
- [ ] Explain the classical baselines, Residual U-Net/hybrid, measured limitations, and a failure case.
- [ ] Add GitHub repository and live application URLs.
