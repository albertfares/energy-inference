# Decomposed Energy Predictor — PyTorch Pipeline

**Status:** Implemented (Phase 1 + Phase 2 + Phase 3 complete)
**Hardware:** Jetson AGX Orin (MAXN)
**Backend:** PyTorch CUDA (live camera pipeline, Logitech C920 webcam, 640×480 YUYV)
**Last updated:** 2026-05-28

---

## TL;DR — Final Results

After three rounds of refinement on top of the original design, the predictor
achieves:

| Metric                                | In-sample MAPE        | Within-family LOOCV MAPE          |
| ------------------------------------- | --------------------- | --------------------------------- |
| End-to-end **FPS**                    | 0.5% (MAE = 0.08 fps) | 0.21% on held-out yolov8n configs |
| End-to-end **Energy / frame**         | 1.1% (MAE = 10.4 mJ)  | 2.4–9.0% on held-out yolov8n configs |
| Capture stage latency                 | 1.9%                  | —                                 |
| Capture stage energy                  | 8.0%                  | —                                 |
| Preprocess stage (lat / E)            | 3.8% / 2.5%           | n/a (per-model; YOLO=0)           |
| Inference stage (lat / E)             | 1.5% / 1.3%           | 5.4–16.7% (per-model)             |
| Postprocess stage (lat / E)           | 1.6% / 14.4%          | n/a (per-model; YOLO=0)           |

**Cross-family generalization** (e.g. predicting SSDLite from yolov8n-only
training) is now caught as an **explicit, named failure** rather than a silent
numerical drift. See [§5.3 Phase 3](#phase-3--per-cv-model-sub-predictors) and
[§9 Known Limitations](#9-known-limitations).

---

## 1. Motivation

The original `train_camera_predictor.py` predictor took
`(model, imgsz, precision, target_fps)` as inputs and directly output
`energy_per_frame`. Two limitations:

1. **FPS was an input, not an output.** In deployment, FPS *emerges* from the
   hardware and the model. Predicting energy at a new operating point required
   guessing FPS first, which is circular.

2. **No stage-level interpretability.** The model couldn't explain how much
   energy came from camera capture vs inference vs idle background, or what
   would change if a component were swapped.

The decomposed design splits the pipeline into independent per-stage
sub-predictors. **FPS and total energy are derived** by combining stage
outputs — neither is an input. Each stage predictor depends only on the
parameters relevant to that stage, so swapping one part of the system
(e.g., model architecture) only requires retraining that stage's predictor.

---

## 2. The PyTorch Pipeline

The live camera inference pipeline (`src/camera_bench/`) is a sequential loop:

```
[1. Capture] → [2. Preprocess] → [3. Inference] → [4. Postprocess] → [throttle sleep] → loop
```

Each stage has been instrumented in `detection.py` via `run_staged_detection()`,
which records `(t_start_ns, t_end_ns)` per stage with `torch.cuda.synchronize()`
between them. `attribute_stage_energy()` in `metrics.py` then maps INA3221
power samples (CPU + GPU + IO rails) onto those timestamps to compute per-stage
energy. This data lives in the `stage_energy.csv` files produced by every
benchmark run, and is flattened into `sweep_summary.csv` columns
(`capture_lat_mean_ms`, `infer_energy_j`, etc.).

The camera is opened via OpenCV's V4L2 backend in **uncompressed YUYV** mode.
The `cap.read()` call returns BGR frames directly — the YUV→BGR conversion is
handled in the V4L2 driver, and there is no JPEG decode step.

---

## 3. Stage Definitions

### Phase 3 stage classification

The decomposed predictor distinguishes between **shared** stages (one
predictor for all models) and **per-model** stages (one predictor block per
CV model, looked up by model name at prediction time):

| Stage       | Shared / Per-model | Why                                                                |
| ----------- | ------------------ | ------------------------------------------------------------------ |
| Capture     | **Shared**         | Depends on camera + resolution; no model-specific behaviour        |
| Preprocess  | **Per-model**      | YOLO fuses it; torchvision models each have their own resize+memcpy cost |
| Inference   | **Per-model**      | Obviously model-dependent                                          |
| Postprocess | **Per-model**      | YOLO fuses it; each torchvision detector has its own NMS cost      |
| Sleep / overhead | **Shared**    | Hardware idle floor; doesn't depend on model                       |

### Stage 1 — Capture (shared)

|                                              |                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------- |
| **What it covers**                           | `cap.read()`: V4L2 blocking wait + USB transfer + YUV→BGR conversion |
| **Naive parameters**                         | Resolution (W × H), target FPS cap                               |
| **Real dependency**                          | Depends on whether the rest of the pipeline runs slower or faster than the camera's frame interval (circular dependency, resolved in the combination layer by Phase 2). |
| **What's predicted (Phase 2)**               | `T_camera_period_ms`, `T_decode_ms`, `E_decode_mj` (per resolution) |
| **What's derived in the combination layer**  | The effective `T_capture` and `E_capture` for the actual configuration |

### Stage 2 — Preprocess (per-model)

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **What it covers** | Resize → normalise → CPU-to-GPU memcpy (only for torchvision models)   |
| **Inputs**       | Input resolution, model input size (`imgsz`), precision                  |
| **YOLO note**    | YOLO fuses this into Stage 3; per-model YOLO predictor returns 0         |

### Stage 3 — Inference (per-model)

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **What it covers** | Model forward pass on CUDA (`model(tensor)`); for YOLO this is fused with pre/post |
| **Inputs**       | `imgsz`, precision (model identity is encoded as the dict key, not a feature) |
| **Note**         | Dominant term at high FPS. Per-model design means generalization risk is bounded to the model whose data the block was trained on. |

### Stage 4 — Postprocess (per-model)

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **What it covers** | NMS + bounding-box decode (CPU) — only for torchvision                 |
| **Inputs**       | `imgsz`                                                                  |
| **YOLO note**    | Fused into Stage 3; per-model YOLO predictor returns 0                   |

### Stage 5 — Overhead / Sleep (shared)

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **What it covers** | Energy consumed while the loop is **sleeping** because of a `target_fps` cap |
| **Predicted**    | One constant: `P_sleep_W` (the idle/wait power draw, ≈ 11 W on MAXN)     |
| **Combination-layer usage** | `E_overhead = P_sleep × max(0, 1/target_fps − T_compute)`         |

---

## 4. Combination Layer (final form, after Phase 1 + 2 + 3)

```python
def predict_pipeline(payload, model, imgsz, precision,
                     target_fps=0, width=640, height=480) -> dict:

    # 1. Per-model lookups for stages 2, 3, 4. Loud failure if model unknown.
    infer_pred       = payload["stage_models"]["infer"     ].get(model)
    preprocess_pred  = payload["stage_models"]["preprocess"].get(model)
    postprocess_pred = payload["stage_models"]["postprocess"].get(model)
    if any(p is None for p in (infer_pred, preprocess_pred, postprocess_pred)):
        raise ValueError(
            f"No predictor registered for model '{model}'. "
            f"Available: {sorted(payload['stage_models']['infer'].keys())}."
        )

    T_pp   , E_pp   = preprocess_pred (imgsz, precision)
    T_inf  , E_inf  = infer_pred      (imgsz, precision)
    T_post , E_post = postprocess_pred(imgsz)
    T_other_ms = T_pp + T_inf + T_post

    # 2. Capture handled by combination layer (Phase 2). Shared camera constants.
    cam = payload["camera_constants"][(width, height)]
    if target_fps > 0:
        T_capture = cam.T_decode_ms                            # throttle covers any wait
    else:
        T_capture = max(cam.T_decode_ms,
                        cam.T_camera_period_ms - T_other_ms)   # camera-limited
    T_wait    = T_capture - cam.T_decode_ms
    E_capture = cam.E_decode_mj + P_sleep_W * T_wait_s * 1000

    # 3. Frame time and FPS
    T_compute = T_capture + T_other_ms
    T_frame   = max(T_compute, 1000.0/target_fps) if target_fps > 0 else T_compute
    fps       = 1000.0 / T_frame

    # 4. Overhead = sleep-only (Phase 1)
    T_sleep    = max(0, T_frame - T_compute)
    E_overhead = P_sleep_W * T_sleep_s * 1000                  # zero at unbounded FPS

    # 5. Total
    E_total = E_capture + E_pp + E_inf + E_post + E_overhead

    return {"fps": fps, "T_frame_ms": T_frame, "E_total_mj": E_total, ...}
```

Three design pieces:

- **`P_sleep × T_sleep` (not `T_frame`)** — stage energies already include
  baseline power during active compute, so multiplying by `T_frame` would
  double-count.
- **Capture handled by the combination layer** — keeps the capture sub-predictor
  a pure function of camera config (no dependency on what model is downstream)
  while still producing correct effective `T_capture`.
- **Per-model lookup with explicit failure** — predicting for an unregistered
  model raises `ValueError` rather than silently extrapolating to a wrong value.

---

## 5. Implementation History — What Was Built vs. What Was Planned

### Phase 0 — Original implementation (followed the plan literally)

Built `scripts/train_decomposed_predictor.py` per the plan:

- Per-stage polynomial Ridge sub-predictors trained on `sweep_summary.csv`
- Capture predictor: features = `(width, height, target_fps)` → outputs
  `T_capture_observed`
- Overhead estimated as `P_overhead = idle_mj / T_frame` (one constant ≈ 5.5 W)
- Combination layer: `E_total = ∑E_stage + P_overhead × T_frame`

**Problems revealed by `scripts/evaluate_decomposed_predictor.py`:**

| Issue                                                            | Symptom                                                  |
| ---------------------------------------------------------------- | -------------------------------------------------------- |
| Overhead **double-counted baseline** at unbounded FPS            | E_total over-predicted by ~30% at target_fps=0           |
| `P_overhead` **mis-estimated** (5.5 W instead of real ~11 W)     | E_total under-predicted by ~30% at low target_fps        |
| Capture stage **conflated decode work + camera wait**            | Capture MAPE 62%; predicted 11 ms for SSDLite when actual was 0.5 ms |

### Phase 1 — Overhead fix (sleep-only, real `P_sleep`)

**Hypothesis:** stage energies from INA3221 already include baseline power.
Overhead should only count energy consumed during the throttle sleep, where
no stage is active.

```python
# OLD
P_overhead = (df.idle_mj/1000) / (1/df.fps_mean)         # ~5.5 W (wrong)
E_overhead = P_overhead * T_frame                         # always > 0

# NEW
throttled  = df[df.target_fps > 0]
T_sleep    = 1/throttled.target_fps - T_compute           # real sleep duration
P_sleep    = (throttled.idle_mj/1000) / T_sleep           # ~11 W (the real value)
E_overhead = P_sleep * max(0, T_frame - T_compute)        # zero at unbounded FPS
```

### Phase 2 — Capture stage decoupling

**Hypothesis:** capture latency depends on whether the model is camera-limited
or compute-limited, which is determined by the *other* stages. Instead of
predicting `T_capture_observed` (model-dependent), predict the **camera's
hardware properties** (`T_camera_period`, `T_decode`, `E_decode`), and let the
combination layer compute the effective capture time.

- New `estimate_camera_constants(df)` derives, per resolution:
  - `T_camera_period_ms` from camera-limited rows: `T_capture_observed + T_other`
  - `T_decode_ms`, `E_decode_mj` from compute-limited rows (where `T_capture < 2 ms`)
- New `compute_capture_outputs(...)` runs in the combination layer:
  - Unbounded FPS, `T_other < period` → `T_capture = period − T_other` (camera-limited)
  - Otherwise → `T_capture = T_decode` (compute-limited or throttle-buffered)
  - `E_capture = E_decode + P_sleep × wait_time`

**Modularity preserved:** the camera constants depend only on
`(width, height)`. Swapping models requires no change to capture.

### Phase 2b — SSDLite `imgsz` semantic fix

The `yolo_imgsz` column for SSDLite rows contains 640 (the camera width), but
`ssdlite320_mobilenet_v3_large` actually hard-codes its input at 320×320
internally. A one-line override in `load_data()` sets `imgsz = 320` for SSDLite
rows so the inference predictor's `imgsz` feature is semantically consistent
across model families.

### Phase 3 — Per-CV-model sub-predictors

**Hypothesis:** the swappable unit in deployment is the **model**, not a
"family" or a shared regressor. The cleanest interpretation of the "lego block"
design is one predictor per CV model. The Phase 2 architecture used one
inference predictor with model one-hot features — which silently degraded when
a model family was missing from training (the LOOCV cross-family ~39% MAPE was
this failure mode).

**Changes:**

- `stage_models["infer"]`, `stage_models["preprocess"]`, `stage_models["postprocess"]`
  are now **dicts keyed by model_short**, e.g.:

  ```python
  stage_models["infer"] = {
      "yolov8n":    {lat_col: predictor, energy_col: predictor},
      "ssdlite320": {lat_col: predictor, energy_col: predictor},
  }
  ```

- Adding a new model = adding a new dict entry. Existing blocks untouched.
- `infer_features()` no longer emits `model_yolov8n` / `model_ssdlite320`
  one-hots; model identity is encoded as the dict key.
- `predict_pipeline()` raises `ValueError` with the list of registered models
  when asked for an unregistered one — no more silent extrapolation through
  zeroed one-hots.
- The LOOCV loop in `loocv_analysis_decomposed_predictor.py` now records
  `status="model_not_in_training"` for folds where the held-out config's
  model has no other examples, instead of running prediction and producing
  silent garbage.

**Capture stays shared.** Sleep power stays shared. These are camera/hardware
properties and don't depend on the model.

---

## 6. Results

All numbers below are from the final Phase 3 run (commit `adcd882`, dataset
unchanged from previous phases).

### 6.1 In-sample accuracy

![Predicted vs Actual end-to-end (in-sample)](figures/decomposed_predictor/pred_vs_actual_scatter.png)

**Figure 1.** Predicted vs measured FPS (left) and energy per frame (right) for
every sweep row, coloured by config. End-to-end MAPE is 0.5% for FPS and 1.1%
for energy across the entire 105-row dataset. The horizontal "ladder" in the
FPS plot reflects the discrete target-FPS grid (5, 10, 15, 20, 25, 30, plus the
two unbounded clusters at ~12 fps for SSDLite and ~30 fps for yolov8n).

### 6.2 Stage breakdown is faithful

![Stage breakdown — predicted vs actual](figures/decomposed_predictor/stage_breakdown_comparison.png)

**Figure 2.** Per-stage contribution to energy (left) and frame latency (right)
for each config at unbounded FPS, with predicted and measured bars side-by-side.
Stack heights and ratios match within a few mJ. SSDLite's energy is dominated
by inference (~1200 mJ); yolov8n energy is split between camera-wait (≈150 mJ)
and inference (≈280 mJ).

### 6.3 Accuracy is uniform across the target-FPS sweep

![Prediction error vs target FPS](figures/decomposed_predictor/error_by_fps.png)

**Figure 3.** FPS error (left) and energy error (right) as a function of
`target_fps`, one curve per config. Every line stays well under the 10% target
band. The slight peaks for SSDLite at unbounded FPS (4.4%) and at high target
FPS (>20) reflect the model being compute-bound at its ~12 FPS ceiling — the
predictor still tracks within ~2-4%.

### 6.4 Per-stage in-sample error

![In-sample stage error heatmap](figures/decomposed_predictor/stage_error_heatmap.png)

**Figure 4.** Mean absolute energy error per stage and per config. The
**preprocess row** for SSDLite is now **2.5%** — in Phase 2 this cell read
**50%** because YOLO's preprocess = 0 was dominating the shared predictor's
data, leaving SSDLite under-fit. Per-model splits resolved this structurally.
Postprocess for SSDLite remains the largest per-stage error (14.4%) but
contributes very little in absolute terms (~5 mJ); end-to-end energy MAPE
stays under 2% on every config.

### 6.5 LOOCV — within-family generalization is clean

LOOCV holds out one `(model, imgsz, precision)` configuration at a time,
retrains all per-model predictors + camera constants + `P_sleep` on the
remaining configs, and predicts the held-out config.

![LOOCV accuracy per held-out config](figures/decomposed_predictor/loocv_config_summary_bar.png)

**Figure 5.** LOOCV FPS MAPE (blue) and energy MAPE (orange) per held-out
config. The four yolov8n configurations all stay under the 10% target
(energy 2.4–9.0%, FPS 0.21%). The SSDLite configuration shows
**"⚠ no predictor"** — when SSDLite is held out, the per-model architecture
correctly reports that no predictor block is registered for `ssdlite320`
instead of fabricating an answer. This is the Phase 3 loud-failure behaviour
working as designed.

### 6.6 LOOCV fragility heatmap

![LOOCV fragility heatmap](figures/decomposed_predictor/fragility_heatmap.png)

**Figure 6.** Mean absolute LOOCV error per stage and per held-out config.
Preprocess and postprocess rows are empty because YOLO has fused stages (both
pred and actual are 0; MAPE is undefined). The SSDLite column is absent
entirely — its hold-out is an explicit failure, not a wrong number. Capture
and inference errors range from 5.4% (best) to 16.7% (worst) within the
yolov8n family; end-to-end energy ranges 2.4–9.0%.

### 6.7 LOOCV predicted vs actual

![LOOCV predicted vs actual](figures/decomposed_predictor/loocv_pred_vs_actual.png)

**Figure 7.** Same scatter format as Figure 1, but with LOOCV predictions on
the four yolov8n configs (84 rows). All points lie tight to the diagonal —
overall LOOCV FPS MAPE 0.2% and energy MAPE 5.5% over the configs the
predictor is qualified to answer for. There is no SSDLite scatter cluster
because those rows were caught as explicit failures upstream.

### 6.8 LOOCV bias

![LOOCV bias chart](figures/decomposed_predictor/loocv_bias_chart.png)

**Figure 8.** Signed mean energy error per held-out config. The four yolov8n
biases sit between −9.0% and +6.0%, with no systematic direction. SSDLite
appears as 0% bias because the explicit-failure folds aren't included in the
numerical aggregate — its status is shown elsewhere.

### 6.9 Number tables (Phase 0 → 1 → 2 → 3)

**Stage-level MAPE on all 105 sweep rows (in-sample):**

| Stage       | Metric  | Phase 0 | Phase 1 | Phase 2 | **Phase 3** |
| ----------- | ------- | ------- | ------- | ------- | ----------- |
| Capture     | Latency | 62.3%   | 62.3%   | 1.9%    | **1.9%**    |
| Capture     | Energy  | 46.2%   | 46.2%   | 8.1%    | **8.0%**    |
| Preprocess  | Latency | 49.9%   | 49.9%   | 49.9% ¹ | **3.8%**    |
| Preprocess  | Energy  | 50.0%   | 50.0%   | 50.0% ¹ | **2.5%**    |
| Inference   | Latency | 1.6%    | 1.6%    | 1.6%    | **1.5%**    |
| Inference   | Energy  | 1.3%    | 1.3%    | 1.3%    | **1.3%**    |
| Postprocess | Latency | 1.6%    | 1.6%    | 1.6%    | **1.6%**    |
| Postprocess | Energy  | 15.0%   | 15.0%   | 15.0%   | **14.4%**   |

¹ In Phase 2 the preprocess MAPE was 50% only because the shared predictor was
swamped by YOLO's preprocess=0 rows, leaving SSDLite's 21 rows with little
signal. Phase 3's per-model architecture resolves this — preprocess MAPE drops
to 3.8%. End-to-end energy is unaffected (the dominant SSDLite preprocess
contribution is only ~20 mJ).

**End-to-end energy MAPE per held-out config (LOOCV):**

| Held-out config          | Phase 0 | Phase 1 | Phase 2 | **Phase 3**                       |
| ------------------------ | ------- | ------- | ------- | --------------------------------- |
| ssdlite320               | 41.0%   | 22.5%   | 38.8%   | **⚠ no predictor (explicit fail)** |
| yolov8n_imgsz320_fp16    | 21.7%   | 46.1%   | 4.5%    | **6.0%**                          |
| yolov8n_imgsz320_fp32    | 16.4%   | 32.0%   | 2.9%    | **4.6%**                          |
| yolov8n_imgsz640_fp16    | 16.6%   | 24.4%   | 2.9%    | **2.4%**                          |
| yolov8n_imgsz640_fp32    | 16.0%   | 27.7%   | 6.2%    | **9.0%**                          |
| **Overall (across configs the predictor can answer)** | — | — | 11.1% | **5.5%** |

The yolov8n LOOCV numbers shift by 1–3 points between Phase 2 and Phase 3 —
the per-model predictor now trains on 63 yolov8n rows instead of 84 mixed rows,
a small loss of statistical power. All four configs still stay under the 10%
target, FPS LOOCV is 0.21% across the board, and the overall energy MAPE
drops from 11.1% to 5.5% because SSDLite's silent-failure 39% no longer
contaminates the aggregate.

### 6.10 Phase comparison summary

The four phases improved different things:

| Phase   | What it fixed                                                | Most-affected metric                                |
| ------- | ------------------------------------------------------------ | --------------------------------------------------- |
| 1       | Overhead double-counting at unbounded FPS                    | E MAPE at `target_fps=0`: 34.2% → 8.2%              |
| 2       | Capture latency conflated decode + camera wait               | Capture in-sample latency MAPE: 62% → 1.9%          |
| 2b      | SSDLite's `imgsz` was 640 (camera width) instead of 320 (model input) | Semantic consistency; in-sample numbers unchanged |
| 3       | Inference predictor silently extrapolated to unseen families | Cross-family LOOCV: silent garbage → explicit failure |

The cumulative effect: in-sample E MAPE 34% → 1.1%; within-family LOOCV E MAPE
~24% → 2.4–9.0%; cross-family LOOCV silent → explicit.

---

## 7. Data Used

The predictor trains on three FPS-sweep runs (all on a 30 fps camera at
640×480 MAXN):

| Sweep directory                                                  | Configs                                | Rows |
| ---------------------------------------------------------------- | -------------------------------------- | ---- |
| `results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_133405/`   | yolov8n imgsz640 fp32 + fp16           | 42   |
| `results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_150803/`   | yolov8n imgsz320 fp32 + fp16           | 42   |
| `results/camera_bench/ssdlite_fps_sweep_MAXN_20260428_162918/`   | ssdlite320 (camera 640; model 320×320 internally) | 21 |

**No new benchmarks were required for any phase.** All four fixes (1, 2, 2b, 3)
were derivable from the existing `sweep_summary.csv` data.

---

## 8. Scripts

| Script                                                | Purpose                                                                                                                                                                                                                              |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `scripts/train_decomposed_predictor.py`               | Trains all per-model sub-predictors, estimates `P_sleep`, derives camera constants. Saves `models/decomposed_predictor.pkl`. Runs per-stage LOOCV and produces 4 diagnostic plots.                                                    |
| `scripts/test_decomposed_predictor.py`                | Interactive CLI for predicting a single config; `--validate` compares to measured rows.                                                                                                                                              |
| `scripts/evaluate_decomposed_predictor.py`            | Comprehensive in-sample evaluation. Saves a detailed CSV with predicted + actual per-stage values, summary tables per config and per target_fps, and the 4 plots used in §6.1–6.4.                                                  |
| `scripts/loocv_analysis_decomposed_predictor.py`      | Leave-one-config-out CV that **retrains all stage models, `P_sleep`, and camera constants per fold**. Records `status="model_not_in_training"` for folds where the per-model predictor has no training data. Produces the plots in §6.5–6.8. |

---

## 9. Known Limitations

### 9.1 Single-family training set

The dataset contains 2 model architectures: yolov8n and ssdlite320. When
LOOCV holds out the only example of a model, **Phase 3 reports
`model_not_in_training` instead of generating a wrong number**. This is the
correct behaviour but it does mean cross-family LOOCV cannot be evaluated
numerically until a second example of each family exists in the dataset.

**Fix:** add at least one more torchvision detector (e.g.
`retinanet_resnet50_fpn` or `fcos_resnet50_fpn`) and re-run the sweep. With
two torchvision examples, the inference predictor for the held-out torchvision
model can be trained on the other one's rows, and cross-family LOOCV becomes
a meaningful number (expected ~5–10% based on within-family results).

### 9.2 Single resolution in training

All sweeps were captured at 640×480. `estimate_camera_constants` produces a
single `(640, 480)` entry. The combination layer falls back to that entry for
unknown resolutions. To support 320×240 or 1280×720, run **Benchmark A** from
the original plan (capture-only, no model) to populate `camera_constants` at
additional resolutions.

### 9.3 YOLO fuses preprocess + inference + postprocess

The Ultralytics YOLO API fuses these stages into a single call. The per-stage
CSV reports preprocess and postprocess as 0 for YOLO; the per-model YOLO
predictors mirror this (return 0). This is faithful to the measurement and
doesn't affect end-to-end accuracy.

---

## 10. Future Work (in priority order)

1. **Add a second torchvision detector** to the sweep — single highest-impact
   change. Validates cross-family generalization and unblocks LOOCV for the
   torchvision family.
2. **Benchmark A (capture-only)** — needed if supporting new resolutions
   becomes a requirement.
3. **Polynomial-degree / regularization tuning** — within-family LOOCV stays
   under 10% but ranges 2.4–9.0%. With more configs per model, simpler models
   (linear in `imgsz²`) or gradient-boosted regression with explicit
   regularization could tighten this further. Low priority since the current
   numbers already meet the target.

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
  figures/decomposed_predictor/         # canonical plot copies for the report
```

---

## 12. One-Line Summary

> **The decomposed predictor achieves 1.1% in-sample energy MAPE and
> 2.4–9.0% within-family LOOCV energy MAPE (0.21% FPS MAPE). Adding a new
> CV model means registering a new per-model block — existing blocks remain
> unchanged, and unregistered models trigger an explicit `ValueError` rather
> than a silent extrapolation.**
