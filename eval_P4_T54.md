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

| Method | RMSE (mm) ↓ | MAE (mm) ↓ | Boundary RMSE (mm) ↓ | PSNR (dB) ↑ | SSIM ↑ | RGB-edge agreement (proxy) ↑ | Boundary accuracy ↑ | Edge F1 ↑ | Latency (ms) | Process RSS (MB) | Python peak (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 212.66 ± 47.29 | 61.72 ± 12.17 | 542.39 ± 116.54 | 20.10 ± 1.50 | 0.823 ± 0.016 | 68.1 ± 4.4% | 10.2 ± 3.1% | 51.8 ± 8.6% | 0.79 ± 0.08 | 259.08 ± 0.09 | 0.33 ± 0.00 |
| Bilinear | 183.15 ± 39.82 | 64.80 ± 14.00 | 467.94 ± 98.68 | 21.38 ± 1.45 | 0.863 ± 0.016 | 73.9 ± 5.2% | 7.1 ± 4.1% | 66.8 ± 2.8% | 0.74 ± 0.07 | 259.08 ± 0.09 | 0.33 ± 0.00 |
| Bicubic | 180.07 ± 39.63 | 65.71 ± 13.85 | 456.99 ± 97.35 | 21.53 ± 1.47 | 0.848 ± 0.017 | 74.8 ± 5.4% | 6.3 ± 3.5% | 58.4 ± 4.3% | 0.74 ± 0.12 | 259.08 ± 0.09 | 0.33 ± 0.00 |
| Joint bilateral | 158.51 ± 37.27 | **52.32 ± 11.67** | 404.49 ± 93.45 | 22.68 ± 1.64 | 0.902 ± 0.015 | 80.2 ± 6.5% | 8.2 ± 4.8% | **68.5 ± 4.1%** | 26.02 ± 6.48 | 259.08 ± 0.09 | 1.36 ± 0.00 |
| Guided filter | **130.30 ± 40.39** | 59.37 ± 23.09 | **307.33 ± 80.68** | **24.51 ± 2.03** | 0.914 ± 0.042 | 90.8 ± 6.2% | 4.8 ± 4.3% | 57.3 ± 4.9% | 1.31 ± 0.15 | 259.08 ± 0.09 | 0.96 ± 0.00 |
| Residual U-Net | 180.19 ± 39.66 | 65.91 ± 13.88 | 457.23 ± 97.45 | 21.53 ± 1.47 | 0.848 ± 0.017 | 74.8 ± 5.4% | 6.3 ± 3.5% | 58.4 ± 4.3% | 10.08 ± 2.06 | 259.08 ± 0.09 | 1.10 ± 0.00 |
| Hybrid (bicubic → guided → U-Net → edge refinement) | 130.47 ± 40.31 | 59.82 ± 23.11 | 307.71 ± 80.58 | 24.50 ± 2.02 | **0.916 ± 0.042** | 90.8 ± 6.2% | 4.8 ± 4.3% | 57.7 ± 5.0% | 12.53 ± 3.05 | 259.08 ± 0.09 | 1.16 ± 0.00 |

## RMSE and SSIM across scales

| Method | 2× RMSE | 2× SSIM | 4× RMSE | 4× SSIM | 8× RMSE | 8× SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Nearest neighbour | 135.13 | 0.888 | 212.66 | 0.823 | 302.30 | 0.738 |
| Bilinear | 120.45 | 0.926 | 183.15 | 0.863 | 263.00 | 0.773 |
| Bicubic | 116.53 | 0.911 | 180.07 | 0.848 | 259.30 | 0.757 |
| Joint bilateral | 92.53 | **0.969** | 158.51 | 0.902 | 243.38 | 0.794 |
| Guided filter | **91.89** | 0.940 | **130.30** | 0.914 | **209.58** | 0.843 |
| Residual U-Net | 116.66 | 0.911 | 180.19 | 0.848 | 259.35 | 0.757 |
| Hybrid | 91.98 | 0.942 | 130.47 | **0.916** | 209.70 | **0.844** |

RMSE is in millimetres. Values are means; complete means and standard deviations for each metric are in `evaluation/results.json`.

## Noisy and missing-depth experiments

Mean RMSE ± standard deviation at 4× across eight synthetic scenes:

| Method | High noise: 36 mm (mm) | 20% missing (mm) |
|---|---:|---:|
| Nearest neighbour | 211.04 ± 41.14 | 238.43 ± 56.99 |
| Bilinear | 181.98 ± 34.80 | 203.97 ± 47.00 |
| Bicubic | 178.82 ± 33.08 | 206.85 ± 49.44 |
| Joint bilateral | 155.24 ± 29.84 | 187.77 ± 48.15 |
| Guided filter | **129.41 ± 37.97** | **146.74 ± 42.33** |
| Residual U-Net | 178.94 ± 33.14 | 206.92 ± 49.42 |
| Hybrid | 129.57 ± 37.91 | 146.91 ± 42.28 |

## Ablation study

4× synthetic held-out scenes; every value is mean ± standard deviation:

| Variant | RMSE (mm) ↓ | MAE (mm) ↓ | Boundary RMSE (mm) ↓ | Boundary MAE (mm) ↓ | SSIM ↑ | Edge F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full hybrid | 130.47 ± 40.31 | 59.82 ± 23.11 | 307.71 ± 80.58 | 251.09 ± 71.64 | 0.916 ± 0.042 | 57.7 ± 5.0% |
| Without guided filter | 180.13 ± 39.66 | 65.58 ± 13.84 | 457.23 ± 97.45 | 328.55 ± 74.60 | 0.850 ± 0.017 | 59.1 ± 4.3% |
| Without U-Net | 130.24 ± 40.38 | 59.39 ± 23.11 | 307.29 ± 80.68 | 251.02 ± 71.71 | 0.916 ± 0.042 | 57.8 ± 5.1% |
| Without edge refinement | 130.53 ± 40.33 | 59.82 ± 23.09 | 307.75 ± 80.58 | 251.07 ± 71.61 | 0.914 ± 0.042 | 57.2 ± 4.8% |
| Without RGB (constant gray guide) | 242.15 ± 50.96 | 117.70 ± 28.62 | 576.73 ± 117.30 | 526.40 ± 116.24 | 0.777 ± 0.031 | 44.3 ± 3.4% |
| Bicubic reference | 180.07 ± 39.63 | 65.71 ± 13.85 | 456.99 ± 97.35 | 328.46 ± 74.58 | 0.848 ± 0.017 | 58.4 ± 4.3% |
| Joint bilateral reference | 158.51 ± 37.27 | **52.32 ± 11.67** | 404.49 ± 93.45 | 275.52 ± 67.63 | 0.902 ± 0.015 | **68.5 ± 4.1%** |

## Interpretation and failure analysis

- The current JSON does not support the requested claim that the hybrid leads 4× RMSE, PSNR, or boundary RMSE: guided filtering is slightly better on those measures (RMSE 130.30 vs 130.47 mm; PSNR 24.512 vs 24.499 dB; boundary RMSE 307.33 vs 307.71 mm). The hybrid has the best 4× SSIM (0.916 vs 0.914).
- At 8×, the hybrid also does not lead RMSE: guided filtering is slightly lower (209.58 vs 209.70 mm). The hybrid has the best 8× SSIM (0.844 vs 0.843).
- Joint bilateral has the best 4× MAE (52.32 mm) and edge F1 (68.5%), and best 2× SSIM (0.969).
- Removing the guided filter increases hybrid RMSE from 130.47 to 180.13 mm, close to the bicubic reference at 180.07 mm. The guided-filter stage contributes the main hybrid RMSE improvement on these scenes.
- Residual U-Net is approximately equal to bicubic at 4× (180.19 vs 180.07 mm). Removing U-Net changes the hybrid RMSE by 0.23 mm (130.47 to 130.24 mm), so it contributes no meaningful RMSE improvement in this evaluation.
- These results are synthetic only. They are not real-sensor validation and must not be compared with NYU or KITTI results.
- Generated scenes do not measure sensor calibration, realistic indoor geometry, thin structures, reflective/transparent surfaces, or RGB/depth misregistration.
- Nearest-valid hole filling can propagate an incorrect measurement through a large missing region. The system does not estimate uncertainty.

## Demo checklist

- [ ] Public application URL works on a first click.
- [ ] Run the generated sample and at least one authorized unseen image.
- [ ] Explain the classical baselines, Residual U-Net/hybrid, measured limitations, and a failure case.
- [ ] Add GitHub repository and live application URLs.
