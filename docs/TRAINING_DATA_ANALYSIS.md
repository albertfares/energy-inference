# Training Data Analysis
> **File:** `data/training_data/filtred_data.csv`
> **Date:** 2026-03-27

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total rows | 909 |
| Columns | 29 |
| Models | 11 |
| Device | CUDA (all rows) |
| Backend | Eager (all rows) |
| Status | All `ok` (pre-filtered) |
| Missing values | None (except `error_msg`, which is empty by design) |

**Sweep dimensions:**

| Dimension | Values |
|---|---|
| Batch size | 1, 2, 4 |
| Resolution | 160, 192, 224, 256, 320, 384, 448, 512, 576, 640 |
| Precision | fp32, fp16, bf16 |

Expected full coverage per model: `3 (batch) × 10 (resolution) × 3 (precision) = 90 rows`.
All models have exactly **90 rows**, except `vit_b_16` which has only **9 rows** — it only ran at resolution 224 (all other resolutions are unsupported, as ViT requires inputs to be exact multiples of its patch size).

---

## 2. Target Variable: `energy_cpu_J`

The primary prediction target.

| Stat | Value |
|---|---|
| Min | 3.67 J |
| Max | 334.63 J |
| Mean | 29.49 J |
| Median | 12.19 J |
| Std | 54.63 J |

The distribution is **heavily right-skewed**: the median (12.19 J) is less than half the mean (29.49 J), driven by SSDLite acting as a strong outlier at the high end. A log transformation of the target may improve predictor performance.

---

## 3. Per-Model Analysis

### 3.1 Energy (CPU) per Model

| Model | Mean (J) | Std | Min | Max | Mean Latency (ms) | Mean FPS |
|---|---|---|---|---|---|---|
| ssdlite | **170.40** | 84.33 | 54.01 | 334.63 | 403.28 | 5.60 |
| swin_t | 28.88 | 20.13 | 16.35 | 133.61 | 65.59 | 41.04 |
| vgg16 | 18.12 | 15.46 | 4.02 | 83.35 | 42.00 | 77.06 |
| vit_b_16 | 17.94 | 7.43 | 10.53 | 34.83 | 41.04 | 54.88 |
| yolo | 13.86 | 1.23 | 11.86 | 15.74 | 31.48 | 74.42 |
| resnet50 | 13.78 | 6.33 | 9.37 | 45.30 | 30.75 | 80.33 |
| googlenet | 12.71 | 1.72 | 10.51 | 20.82 | 28.43 | 81.97 |
| shufflenet_v2_x1_0 | 11.76 | 1.15 | 9.97 | 13.98 | 26.27 | 89.29 |
| mobilenet_v3_large | 11.42 | 1.21 | 9.58 | 15.60 | 25.51 | 91.65 |
| mobilenet_v3_small | 9.57 | 1.03 | 7.95 | 11.64 | 21.55 | 109.09 |
| resnet18 | **5.59** | 2.34 | 3.67 | 15.73 | 12.04 | 201.01 |

**Key observations:**

- **SSDLite is a strong outlier**: its mean CPU energy (170 J) is ~6× the next highest model (Swin-T at 29 J). This is driven by detection overhead — SSDLite runs NMS and multi-scale head computation on top of its backbone, and its latency (403 ms) is nearly 10× that of ResNet18. This skews the overall dataset distribution significantly.
- **ResNet18 is the most efficient classifier**, with a mean of 5.59 J and latency of 12 ms.
- **MobileNetV3-Small** is the most efficient among models designed for mobile (9.57 J), consistent with its design goal.
- **YOLO** shows surprisingly low and stable energy (13.86 J, std=1.23) for a detection model — much lower than SSDLite.
- **VGG16 and ViT-B/16** have comparable mean energy (~18 J), but for very different reasons: VGG is heavy due to its large FC layers, while ViT's limited data (only 9 rows at batch 1/2/4) may underrepresent its true energy profile at larger resolutions.
- **Swin-T** is the most energy-hungry transformer after ViT, with high variance — it scales notably with batch.

---

## 4. Effect of Sweep Parameters on Energy

### 4.1 Batch Size

| Batch | Mean Energy (J) | Std |
|---|---|---|
| 1 | 18.91 | 22.49 |
| 2 | 27.06 | 42.41 |
| 4 | 42.51 | 79.88 |

Energy increases with batch, but the effect is **highly model-dependent**:

| Model | Batch=1 | Batch=2 | Batch=4 | Scaling pattern |
|---|---|---|---|---|
| ssdlite | 85.30 | 152.93 | 272.98 | Near-linear |
| swin_t | 20.88 | 25.69 | 40.07 | Sub-linear |
| vgg16 | 9.65 | 16.29 | 28.43 | Near-linear |
| vit_b_16 | 11.03 | 17.90 | 24.89 | Sub-linear |
| resnet50 | 11.23 | 12.44 | 17.67 | Sub-linear |
| resnet18 | 4.41 | 5.32 | 7.03 | Sub-linear |
| yolo | 13.69 | 13.91 | 13.98 | **Flat** |
| mobilenet_v3_large | 11.33 | 11.25 | 11.69 | **Flat** |
| mobilenet_v3_small | 9.61 | 9.46 | 9.65 | **Flat** |
| shufflenet_v2_x1_0 | 11.62 | 11.80 | 11.86 | **Flat** |
| googlenet | 12.21 | 12.40 | 13.51 | Near-flat |

YOLO, MobileNetV3, and ShuffleNet are **almost insensitive to batch size** in CPU energy — suggesting their bottleneck is not compute but memory or pipeline latency. SSDLite and VGG scale more aggressively.

### 4.2 Resolution

| Resolution | Mean Energy (J) |
|---|---|
| 160 | 27.22 |
| 224 | 26.55 |
| 320 | 28.33 |
| 448 | 28.25 |
| 512 | 31.68 |
| 576 | 33.68 |
| 640 | 37.22 |

Resolution has a **weak overall effect** (Pearson r = 0.054 with `energy_cpu_J`). Energy stays relatively flat between 160–448, then climbs slightly from 512 onward. This is likely masked by SSDLite's flat response to resolution (its energy is dominated by the detection head, not spatial resolution), which dilutes the correlation.

### 4.3 Precision (fp32 vs fp16 vs bf16)

| Precision | Mean Energy (J) | Std |
|---|---|---|
| fp32 | 29.76 | 54.25 |
| fp16 | 29.53 | 56.14 |
| bf16 | 29.19 | 53.64 |

At the aggregate level, precision has **almost no impact** on CPU energy. This is expected on CUDA: fp16/bf16 benefits manifest mainly in GPU throughput and memory, not directly in CPU-side energy (which covers data movement, framework overhead, and host-side operations).

Per-model breakdown reveals some nuance:

| Model | fp32 | fp16 | bf16 | Trend |
|---|---|---|---|---|
| swin_t | 36.92 | 24.94 | 24.79 | fp32 >> reduced |
| vit_b_16 | 22.06 | 15.88 | 15.87 | fp32 >> reduced |
| vgg16 | 21.90 | 16.14 | 16.33 | fp32 > reduced |
| mobilenet_v3_small | 8.21 | 10.05 | 10.45 | fp32 < reduced (reversed!) |
| yolo | 12.23 | 14.31 | 15.04 | fp32 < reduced (reversed!) |

Transformers (Swin, ViT) benefit clearly from reduced precision — the large matrix multiplications in attention heads are expensive in fp32. Some smaller models (MobileNet, YOLO) show the opposite trend, possibly because reduced precision adds overhead on operations that are already small.

---

## 5. Correlation Analysis

Pearson correlations with `energy_cpu_J`:

| Feature | Correlation |
|---|---|
| `latency_ms` | **+0.982** |
| `energy_gpu_J` | +0.856 |
| `batch` | +0.179 |
| `resolution` | +0.054 |
| `macs_total` | +0.007 |
| `flops_per_sample` | -0.038 |
| `num_params` | -0.091 |
| `fps` | -0.377 |
| `power_total_W` | -0.220 |

**Critical finding: `latency_ms` alone explains ~96% of the variance in `energy_cpu_J`** (r² ≈ 0.96). This is physically expected — CPU energy is approximately `power × time`, and if average power is roughly constant, energy tracks latency almost perfectly.

However, this is not directly useful for a *predictive* model, because you can't know latency without running the benchmark. The predictor must estimate energy from *static* features (model architecture, batch, resolution, precision) — and the correlations of those with energy are all weak:

- `macs_total` (r = 0.007) and `num_params` (r = -0.091) are **near-zero** — FLOPs alone do not predict energy.
- `power_total_W` has a **negative** correlation with energy: high-power models (VGG: 32W, ViT: 28W) are not the highest energy ones, while SSDLite draws low power (11.68W) but runs for a very long time, accumulating the most energy.

This confirms that the predictor must capture **model identity** (family, architecture type) as a feature, not just hardware-level metrics.

---

## 6. Notable Anomalies and Data Quality Notes

### SSDLite dominance
SSDLite accounts for most of the tail of the energy distribution. Its energy values (54–334 J) are so far above the rest that any regressor trained on the full dataset may be heavily influenced by it. Consider:
- Training a **separate detector regressor** (already supported via `--separate-by model_task`)
- Or applying **log-transform** on `energy_cpu_J`

### ViT-B/16 underrepresentation
With only 9 rows (all at resolution 224), ViT-B/16 is severely underrepresented compared to other models (90 rows each). Predictions for ViT at any batch/precision combination will extrapolate with high uncertainty. This model should either be **excluded from training** or clearly flagged in predictor outputs.

### YOLO and MobileNet batch-insensitivity
These models' CPU energy is almost flat across batch sizes. This is a real signal, not noise — the predictor should learn it, which is why `model` identity as a feature matters more than `macs_total`.

### Negative `power_total_W` correlation
This is non-obvious: models that draw more instantaneous power (VGG, ViT) do not necessarily consume more total energy, because they may finish faster. SSDLite draws only 11.68 W on average but runs for 403 ms — resulting in the highest energy. This power × time interplay is key.

---

## 7. Summary of Key Findings

1. **Latency is the best proxy for energy** (r=0.982), but cannot be used as a predictor input (it requires running the model).
2. **FLOPs and parameter count do not predict energy** — the relationship is nearly zero after controlling for model identity.
3. **SSDLite is a structural outlier** — its energy profile is 6× the next model. Separate modeling for detectors is justified.
4. **Batch size has a heterogeneous effect**: scales energy ×2–3 for SSDLite and VGG; nearly no effect for YOLO and MobileNet families.
5. **Precision has minimal impact on CPU energy** overall, with exceptions for attention-heavy models (Swin-T, ViT-B/16) where fp32 is noticeably costlier.
6. **Resolution has weak effect** on energy up to ~448px, with a gradual increase beyond 512px.
7. **Model identity (family/architecture) is a necessary feature** for any energy predictor — raw hardware metrics alone are insufficient.
8. **ViT-B/16 coverage gap** (9 rows vs 90 for others) should be treated as a known limitation.
