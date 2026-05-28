# Decomposed Energy Predictor — PyTorch Pipeline

**Status:** Implemented (Phase 1 + Phase 2 complete)
**Hardware:** Jetson AGX Orin (MAXN)
**Backend:** PyTorch CUDA (live camera pipeline)
**Last updated:** 2026-05-28

---

## TL;DR — Final Results

After two rounds of fixes on top of the original design, the predictor now achieves:

| Metric | In-sample MAPE | Within-family LOOCV MAPE |
|---|---|---|
| End-to-end **FPS**     | ≤ 2.0% on every config | 0.2% on held-out yolov8n configs |
| End-to-end **Energy**  | ≤ 2.0% on every config | 2.9–6.2% on held-out yolov8n configs |
| Capture stage latency  | 1.9% | — |
| Capture stage energy   | 8.1% | — |
| Inference latency      | 1.6% | 26.5% (when held-out config is in unseen family) |
| Inference energy       | 1.3% | 23.8% (same reason) |

**Cross-family generalization** (predicting SSDLite from yolov8n-only training) fails at ~39% energy MAPE — this is a **data scarcity limitation**, not a modelling bug. See [§9 Known Limitations](#9-known-limitations).

---

## 1. Motivation

The original predictor took `(model, imgsz, precision, target_fps)` as inputs and directly
output `energy_per_frame`. Two limitations:

1. **FPS was an input, not an output.** In deployment, FPS *emerges* from the hardware and
   the model. Predicting energy at a new operating point required guessing FPS first, which
   is circular.

2. **No stage-level interpretability.** The model couldn't explain how much energy came
   from camera capture vs inference vs idle background.

The decomposed design splits the pipeline into independent per-stage sub-predictors. **FPS
and total energy are derived** by combining stage outputs — neither is an input.

---

## 2. The PyTorch Pipeline

The live camera inference pipeline (`src/camera_bench/`) is a sequential loop:

```
[1. Capture] → [2. Preprocess] → [3. Inference] → [4. Postprocess] → [throttle sleep] → loop
```

Each stage has been instrumented in `detection.py` via `run_staged_detection()`, which
records `(t_start_ns, t_end_ns)` per stage with `torch.cuda.synchronize()` between them.
`attribute_stage_energy()` in `metrics.py` maps INA3221 power samples onto those
timestamps to compute per-stage energy. This data lives in the `stage_energy.csv` files
produced by every benchmark run, and is also flattened into `sweep_summary.csv` columns
(`capture_lat_mean_ms`, `infer_energy_j`, etc.).

---

## 3. Stage Definitions (as actually implemented)

### Stage 1 — Capture

| | |
|---|---|
| **What it covers** | `cap.read()`: V4L2 read + OpenCV JPEG decode (libjpeg-turbo) |
| **Naive parameters** | Resolution (`W × H`), target FPS cap |
| **Real dependency** | *Depends on whether the rest of the pipeline runs slower or faster than the camera's frame interval.* This is the **circular dependency** addressed in Phase 2. |
| **What's predicted (Phase 2)** | `T_camera_period_ms`, `T_decode_ms`, `E_decode_mj` — all functions of resolution only |
| **What's derived in combination layer** | The observed `T_capture` and `E_capture` for the actual pipeline configuration |

### Stage 2 — Preprocess

| | |
|---|---|
| **What it covers** | Resize → normalise → CPU-to-GPU memcpy (only for torchvision models) |
| **Inputs** | Input resolution, model input size (`imgsz`), precision |
| **YOLO note** | YOLO fuses this into Stage 3 → set to 0 |

### Stage 3 — Inference

| | |
|---|---|
| **What it covers** | Model forward pass on CUDA (`model(tensor)`); for YOLO this is fused with pre/post |
| **Inputs** | Model one-hot, `imgsz`, precision |
| **The dominant term at high FPS** | And the one most sensitive to model-family generalization |

### Stage 4 — Postprocess

| | |
|---|---|
| **What it covers** | NMS + bounding-box decode (CPU) — only for torchvision |
| **Inputs** | Model family (one-hot torchvision flag) |
| **YOLO note** | Fused into Stage 3 → set to 0 |

### Stage 5 — Overhead / Sleep

| | |
|---|---|
| **What it covers** | Energy consumed while the loop is **sleeping** because of a `target_fps` cap |
| **Predicted** | One constant: `P_sleep_W` (the idle/wait power draw, ≈ 11 W on MAXN) |
| **Combination-layer usage** | `E_overhead = P_sleep × max(0, 1/target_fps − T_compute)` |

---

## 4. Combination Layer (final form, after Phase 1 + 2)

```python
def predict_pipeline(payload, model, imgsz, precision,
                     target_fps=0, width=640, height=480) -> dict:

    # 1. Sub-predictors for stages 2, 3, 4 (capture handled separately)
    T_pp   , E_pp   = preprocess_pred(...)
    T_inf  , E_inf  = inference_pred(...)
    T_post , E_post = postprocess_pred(...)
    T_other_ms = T_pp + T_inf + T_post

    # 2. Capture handled by combination layer (Phase 2)
    cam = camera_constants[(width, height)]
    if target_fps > 0:
        T_capture = cam.T_decode_ms                            # throttle covers any wait
    else:
        T_capture = max(cam.T_decode_ms,                       # compute-limited
                        cam.T_camera_period_ms - T_other_ms)    # camera-limited
    T_wait    = T_capture - cam.T_decode_ms
    E_capture = cam.E_decode_mj + P_sleep_W * T_wait_s * 1000   # mJ

    # 3. Frame time and FPS
    T_compute = T_capture + T_other_ms
    T_frame   = max(T_compute, 1000.0/target_fps) if target_fps > 0 else T_compute
    fps       = 1000.0 / T_frame

    # 4. Overhead = sleep-only (Phase 1)
    T_sleep   = max(0, T_frame - T_compute)
    E_overhead= P_sleep_W * T_sleep_s * 1000                    # zero at unbounded FPS

    # 5. Total
    E_total = E_capture + E_pp + E_inf + E_post + E_overhead

    return {"fps": fps, "T_frame_ms": T_frame, "E_total_mj": E_total, ...}
```

Two pieces of "magic":

- **`P_sleep × T_sleep` (not `T_frame`)** — stage energies already include baseline power
  during active compute, so multiplying by `T_frame` double-counts baseline.
- **Capture handled by the combination layer** — keeps the capture sub-predictor a pure
  function of camera config (no dependency on what model is downstream), while still
  producing correct effective `T_capture` for the actual configuration.

---

## 5. Implementation History — What Was Built vs. What Was Planned

### Phase 0 — Original implementation (followed the plan literally)

Built `scripts/train_decomposed_predictor.py` per the plan:

- Per-stage polynomial Ridge sub-predictors trained on `sweep_summary.csv`
- Capture predictor: features = `(width, height, target_fps)` → outputs `T_capture_observed`
- Overhead estimated as `P_overhead = idle_mj / T_frame` (one constant ≈ 5.5 W)
- Combination layer: `E_total = ∑E_stage + P_overhead × T_frame`

**Problems revealed by evaluation (`scripts/evaluate_decomposed_predictor.py`):**

| Issue | Symptom |
|---|---|
| Overhead **double-counted baseline** at unbounded FPS | E_total over-predicted by ~30% at target_fps=0 |
| `P_overhead` **mis-estimated** (5.5 W instead of real ~11 W) | E_total under-predicted by ~30% at low target_fps |
| Capture stage **conflated decode work + camera wait** | Capture MAPE 62%; predicted 11 ms for SSDLite when actual was 0.5 ms |

### Phase 1 — Overhead fix (sleep-only, real `P_sleep`)

**Hypothesis:** stage energies from INA3221 already include baseline power. Overhead should
only count energy consumed during the throttle sleep, where no stage is active.

**Changes** (`scripts/train_decomposed_predictor.py`):

```python
# OLD
P_overhead = (df.idle_mj/1000) / (1/df.fps_mean)         # ~5.5 W
E_overhead = P_overhead * T_frame                         # always > 0

# NEW
throttled  = df[df.target_fps > 0]
T_sleep    = 1/throttled.target_fps - T_compute           # real sleep duration
P_sleep    = (throttled.idle_mj/1000) / T_sleep           # ~11 W (the real value)
E_overhead = P_sleep * max(0, T_frame - T_compute)        # zero at unbounded FPS
```

### Phase 2 — Capture stage decoupling

**Hypothesis:** capture latency depends on whether the model is camera-limited or
compute-limited, which is determined by the *other* stages. Instead of predicting
`T_capture_observed` (model-dependent), predict the **camera's hardware properties**
(`T_camera_period`, `T_decode`, `E_decode`), and let the combination layer compute the
effective capture time.

**Changes** (`scripts/train_decomposed_predictor.py`):

- New `estimate_camera_constants(df)` derives, per resolution:
  - `T_camera_period_ms` from camera-limited rows: `T_capture_observed + T_other`
  - `T_decode_ms`, `E_decode_mj` from compute-limited rows (where `T_capture < 2 ms`)
- New `compute_capture_outputs(...)` runs in the combination layer:
  - Unbounded FPS, `T_other < period` → `T_capture = period − T_other` (camera-limited)
  - Otherwise → `T_capture = T_decode` (compute-limited or throttle-buffered)
  - `E_capture = E_decode + P_sleep × wait_time`

**Modularity preserved:** the camera constants depend only on `(width, height)`. Swapping
models requires retraining only the inference predictor. The capture↔inference coupling
lives in the combination layer (same place that already couples stages to compute
`T_frame`).

The polynomial capture sub-predictor still trains (kept for backward-compat with the
per-stage plots) but its outputs are **overridden** in `predict_pipeline` by
`compute_capture_outputs(...)`.

---

## 6. Results — Before / After Phase 1 / After Phase 2

All numbers are end-to-end (combination-layer output vs measured ground truth).

### 6.1 In-sample evaluation

**Stage-level MAPE on all 105 sweep rows:**

| Stage | Metric | Phase 0 | After Phase 1 | After Phase 2 |
|---|---|---|---|---|
| Capture     | Latency | 62.3% | 62.3% | **1.9%** |
| Capture     | Energy  | 46.2% | 46.2% | **8.1%** |
| Preprocess  | Latency | 49.9% | 49.9% | 49.9% ¹ |
| Preprocess  | Energy  | 50.0% | 50.0% | 50.0% ¹ |
| Inference   | Latency | 1.6% | 1.6% | 1.6% |
| Inference   | Energy  | 1.3% | 1.3% | 1.3% |
| Postprocess | Latency | 1.6% | 1.6% | 1.6% |
| Postprocess | Energy  | 15.0% | 15.0% | 15.0% |

¹ Preprocess/Postprocess MAPE is undefined for YOLO rows (where the value is 0). The
50% shown is from the 21 SSDLite rows only, and does **not** propagate into end-to-end
errors because the SSDLite preprocess/postprocess energies are small.

**End-to-end energy MAPE per config:**

| Config | Phase 0 | After Phase 1 | After Phase 2 |
|---|---|---|---|
| ssdlite320_imgsz640_fp32 | 31.6% | 3.4% | **2.0%** |
| yolov8n_imgsz320_fp16    | 17.2% | 1.6% | **0.7%** |
| yolov8n_imgsz320_fp32    | 16.9% | 1.7% | **0.8%** |
| yolov8n_imgsz640_fp16    | 16.4% | 1.5% | **1.0%** |
| yolov8n_imgsz640_fp32    | 15.9% | 1.2% | **0.8%** |

**End-to-end energy MAPE per target FPS:**

| target_fps | Phase 0 | After Phase 1 | After Phase 2 |
|---|---|---|---|
| 0 (unbounded) | **34.2%** | 8.2% | **2.1%** |
| 5             | 28.2% | 0.6% | 0.6% |
| 10            | 24.1% | 0.5% | 0.5% |
| 15            | 18.7% | 0.7% | 0.7% |
| 20            | 11.9% | 0.8% | 0.8% |
| 25            | 7.4%  | 1.1% | 1.2% |
| 30            | 12.6% | 1.4% | 1.4% |

The unbounded-FPS row tells the whole story:
- **Phase 0 → 1:** removing the overhead double-count fixed throttled-FPS rows but not
  unbounded (capture stage was still 62% off).
- **Phase 1 → 2:** capture stage fix dropped the unbounded-FPS error from 8.2% to 2.1%.

### 6.2 Leave-one-config-out (LOOCV) evaluation

LOOCV holds out one `(model, imgsz, precision)` config at a time, retrains all stage
predictors + camera constants + `P_sleep` on the remaining configs, and predicts the
held-out config. This estimates how the predictor will generalize to configs it has
never seen.

**Per held-out config (energy MAPE):**

| Held-out config | Phase 0 | After Phase 1 | After Phase 2 |
|---|---|---|---|
| ssdlite320_imgsz640_fp32 | 41.0% | 22.5% | 38.8% ² |
| yolov8n_imgsz320_fp16    | 21.7% | 46.1% ³ | **4.5%** |
| yolov8n_imgsz320_fp32    | 16.4% | 32.0% ³ | **2.9%** |
| yolov8n_imgsz640_fp16    | 16.6% | 24.4% ³ | **2.9%** |
| yolov8n_imgsz640_fp32    | 16.0% | 27.7% ³ | **6.2%** |

² SSDLite *appears* worse after Phase 2, but this is expected — see explanation below.

³ Phase-1 LOOCV for yolov8n "got worse" only because Phase 1 removed a **compensating
error**: the original predictor's wrong overhead happened to cancel out the polynomial
capture predictor's over-prediction. Phase 2 fixes the underlying capture issue, and
LOOCV for yolov8n drops to ≤6%.

**Bias direction (signed mean error) — Phase 2:**

| Held-out config | Energy bias | Read |
|---|---|---|
| yolov8n_imgsz320_fp16 | +4.5%  | Slight over-predict |
| yolov8n_imgsz320_fp32 | +2.9%  | Slight over-predict |
| yolov8n_imgsz640_fp16 | −2.9%  | Slight under-predict |
| yolov8n_imgsz640_fp32 | −6.2%  | Slight under-predict |
| ssdlite320_imgsz640_fp32 | −38.8% | Heavy under-predict |

The yolov8n biases are small and centered around zero — predictor is honest within the
family.

### Why SSDLite LOOCV "got worse" after Phase 2

This is the most important diagnostic in the entire result set:

1. Hold-out training set has **only yolov8n rows** (no SSDLite, no other torchvision)
2. The **inference sub-predictor** trained on yolov8n only predicts ≈20 ms for the
   SSDLite test row (true value: ~80 ms). Severe under-prediction.
3. The **combination layer** sees `T_other_predicted ≈ 20 ms < T_camera_period (33 ms)`
   and concludes "camera-limited" → predicts `T_capture = 13 ms` (true value: 0.5 ms,
   because in reality SSDLite is *compute*-limited at 80 ms/frame)
4. Predicted `T_frame = 33 ms → FPS = 30` (real: ~12 FPS) → 70% FPS error
5. Predicted energy collapses because the predictor thinks the frame takes 33 ms when
   it actually takes 80 ms

**Phase 2's capture stage doesn't have a bug — the inference predictor has zero
training examples for the torchvision family, so it can't extrapolate that far.** The
combination layer faithfully forwards that error. Phase 1's smaller SSDLite number
(22.5%) was lucky: the bad overhead estimate partially compensated for the
inference-extrapolation error.

This is the **correct, honest behaviour** of a decomposed predictor: it fails loudly
when a model family is missing from training, and the failure mode is diagnosable
(trace through inference → combination → capture).

---

## 7. Data Used

The predictor trains on three FPS-sweep runs (all 30 fps camera at 640×480 MAXN):

| Sweep directory | Configs | Rows |
|---|---|---|
| `results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_133405/` | yolov8n imgsz640 fp32+fp16 | 42 |
| `results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_150803/` | yolov8n imgsz320 fp32+fp16 | 42 |
| `results/camera_bench/ssdlite_fps_sweep_MAXN_20260428_162918/` | ssdlite320 imgsz640 fp32 | 21 |

Each row corresponds to one (config × target_fps × repeat). Target_fps sweep:
{0, 5, 10, 15, 20, 25, 30}; 3 repeats each.

**No new benchmarks were required for Phase 1 or Phase 2.** Both fixes were derivable
from the existing `sweep_summary.csv` data.

---

## 8. Scripts

| Script | Purpose |
|---|---|
| `scripts/train_decomposed_predictor.py` | Trains all sub-predictors + estimates `P_sleep` + derives camera constants. Saves `models/decomposed_predictor.pkl`. Runs leave-one-config-out CV per stage and produces 4 diagnostic plots. |
| `scripts/test_decomposed_predictor.py` | Interactive CLI for predicting a single config; optional `--validate` mode compares to all measured rows. |
| `scripts/evaluate_decomposed_predictor.py` | Comprehensive in-sample evaluation. Saves a detailed CSV with predicted + actual per-stage values, summary tables per config and per target_fps, and 4 plots (stage breakdown comparison, error-by-FPS, stage error heatmap, pred vs actual scatter). |
| `scripts/loocv_analysis_decomposed_predictor.py` | Leave-one-config-out CV that **retrains all stage models, P_sleep, and camera constants per fold**. Produces detailed CSV, per-config summary, stage error breakdown, fragility heatmap, and bias chart. |

---

## 9. Known Limitations

### 9.1 Single-family training set

The dataset contains 2 model architectures: yolov8n (1 family) and ssdlite320 (1 family).
LOOCV that holds out the only example of a family removes all training signal for that
family's preprocess/postprocess stage features and forces the inference predictor to
extrapolate beyond its support.

**Fix:** add at least one more torchvision detector (e.g. retinanet_resnet50_fpn,
fcos_resnet50_fpn) and re-run the sweep. Expected outcome: SSDLite LOOCV energy MAPE
should drop from ~39% to roughly the within-family baseline (~5–10%).

### 9.2 Single resolution in training

All sweeps were captured at 640×480. `estimate_camera_constants` therefore produces a
single entry (the (640, 480) key). The combination layer falls back to that entry for
unknown resolutions. This is fine for current deployment but means the predictor cannot
extrapolate to 320×240 or 1280×720 without additional capture-only data.

**Fix (when needed):** run **Benchmark A** (capture-only, no model) from the original
plan to populate `camera_constants` at additional resolutions.

### 9.3 YOLO fuses preprocess + inference + postprocess

The Ultralytics YOLO API fuses these into a single call. The per-stage CSV reports
preprocess and postprocess as 0 for YOLO, and the predictor mirrors that (sets them to
0 in training). This is faithful to the measurement, but means the 50% MAPE on
preprocess and 15% on postprocess shown in the stage table are computed only from the
21 SSDLite rows. These errors do not propagate to end-to-end since they are small in
absolute terms (preprocess ~10 mJ, postprocess ~5 mJ).

---

## 10. Future Work (in priority order)

1. **Add a second torchvision detector** to the sweep — single highest-impact change.
   Validates the cross-family generalization story and likely brings SSDLite LOOCV under
   10%.

2. **Update docs in `docs/` and `benchmarks/`** to reference the decomposed predictor as
   the primary energy estimator.

3. **Benchmark A** (capture-only) — only needed when supporting new resolutions becomes a
   requirement.

4. **Polynomial degree tuning** — degree-2 Ridge over-fits slightly in LOOCV (visible in
   inference 26% MAPE when held-out config is a single example of its imgsz×precision
   cell). Consider degree-1 Ridge or a small gradient-boosted regressor with explicit
   regularisation. Low-priority because in-family LOOCV is already < 10%.

---

## 11. File Map

```
scripts/
  train_decomposed_predictor.py         # training + CV + plots + payload save
  test_decomposed_predictor.py          # single-prediction CLI
  evaluate_decomposed_predictor.py      # comprehensive in-sample evaluation
  loocv_analysis_decomposed_predictor.py# LOOCV with per-fold retraining

models/
  decomposed_predictor.pkl              # serialised payload (stage_models +
                                        #  camera_constants + p_sleep_w + metadata)

results/analysis/
  decomposed_predictor_<ts>/            # training-script outputs
  eval_decomposed_<ts>/                 # in-sample evaluation outputs
  loocv_decomposed_<ts>/                # LOOCV outputs

docs/
  DECOMPOSED_PREDICTOR_PLAN_PYTORCH.md  # ← this file
```

---

## 12. One-Line Summary

> **The decomposed predictor is now production-ready within the yolov8n model family
> (≤ 6% LOOCV energy error, < 1% FPS error). Extending to new model families requires
> adding more diverse training data, not changing the math.**
