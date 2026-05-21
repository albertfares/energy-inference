# Decomposed Energy Predictor — PyTorch Pipeline

**Status:** Planning  
**Hardware:** Jetson AGX Orin (MAXN)  
**Backend:** PyTorch CUDA (live camera pipeline)

---

## 1. Motivation

The current predictor takes `(model, imgsz, precision, target_fps)` as inputs and directly
outputs `energy_per_frame`. This has two limitations:

1. **FPS is an input, not an output.** In deployment you don't choose FPS — it emerges from
   the hardware and the model. Predicting energy at a new operating point requires guessing
   FPS first, which is circular.

2. **No stage-level interpretability.** You cannot tell how much of the energy comes from the
   camera vs the model inference, or reason about what changes if you swap a component.

The new design splits the pipeline into **independent stages**. Each stage has its own
sub-predictor that only depends on parameters relevant to that stage. **FPS and total energy
are derived** by combining the stage outputs — neither is an input.

---

## 2. The PyTorch Pipeline

The live camera inference pipeline in `src/camera_bench/` has the following stages:

```
[1. Capture]  →  [2. Preprocess]  →  [3. Inference]  →  [4. Postprocess]  →  [5. Overhead]
```

Each stage has already been instrumented in `detection.py` via `run_staged_detection()`,
which records `(t_start_ns, t_end_ns)` per stage with `torch.cuda.synchronize()` between
them. The `attribute_stage_energy()` function in `metrics.py` then maps INA3221 power
samples onto those timestamps to compute per-stage energy. **This data already exists in
the `stage_energy.csv` files produced by every benchmark run.**

---

## 3. Stage Definitions

### Stage 1 — Capture

| | |
|---|---|
| **What it covers** | `cap.read()`: V4L2 read + OpenCV JPEG decode (libjpeg-turbo) |
| **Key parameters** | Resolution (`W × H`), target FPS cap |
| **Independent of** | Model, precision |
| **Notes** | At low target FPS the camera waits between frames; `T_capture` includes that idle time. At unbounded FPS, `T_capture = 1 / camera_hardware_max_fps`. |

### Stage 2 — Preprocess

| | |
|---|---|
| **What it covers** | Resize, normalise (CPU), `.to(device)` CUDA memcpy |
| **Key parameters** | Input resolution (`W × H`), model input size (`imgsz`), precision |
| **Independent of** | Model architecture |
| **Notes** | Dominated by the CPU→GPU memcpy at high resolution. Scales roughly with pixel count. |

### Stage 3 — Inference

| | |
|---|---|
| **What it covers** | Model forward pass on CUDA (`model(tensor)`) |
| **Key parameters** | Model (`yolov8n`, `ssdlite320`, …), `imgsz`, `precision` (FP16/FP32) |
| **Independent of** | Camera resolution, capture stage |
| **Notes** | The dominant energy term at high FPS. YOLO wraps pre+infer+post into a single fused call; torchvision models expose each separately. |

### Stage 4 — Postprocess

| | |
|---|---|
| **What it covers** | NMS, bounding box decode, confidence filter (CPU) |
| **Key parameters** | Model family (torchvision vs YOLO), number of detections |
| **Independent of** | Camera resolution, precision |
| **Notes** | Small for most models. YOLO fuses this into Stage 3 — `T_postprocess = 0` for YOLO runs. |

### Stage 5 — Overhead

| | |
|---|---|
| **What it covers** | Python loop bookkeeping, INA3221 polling, result buffering, sleep (when throttled) |
| **Key parameters** | Target FPS (determines sleep time) |
| **Notes** | Roughly constant power draw. Becomes the dominant **energy** term at very low FPS because it runs for the full frame interval even when inference is fast. |

---

## 4. How FPS and Total Energy Are Derived

### 4.1 Time model

The stages run **sequentially** in the PyTorch pipeline (no parallel pipeline like DeepStream):

```
T_frame = T_capture + T_preprocess + T_infer + T_postprocess
FPS     = 1 / T_frame
```

If a `target_fps` cap is set, the loop sleeps for `max(0, 1/target_fps - T_frame)` seconds.
The effective frame time is then `max(T_frame, 1/target_fps)`.

> **Note:** FPS is now a **prediction output**, not an input. Given a model and resolution,
> you sum the stage times to get `T_frame`, and FPS follows. No sweep needed.

### 4.2 Energy model

All stages consume energy simultaneously (the CPU is always running even during GPU inference):

```
E_total_per_frame = E_capture + E_preprocess + E_infer + E_postprocess + E_overhead
```

`E_overhead` is a fixed power draw multiplied by frame time:

```
E_overhead_per_frame = P_overhead × T_frame
```

This naturally explains the observed result: **at low target FPS, energy per frame is high**
because `T_frame = 1/target_fps` is large and `E_overhead` grows proportionally, even though
inference itself takes the same time and energy.

---

## 5. Data We Already Have

The existing PyTorch FPS sweeps (`results/camera_bench/yolov8n_fps_sweep_*`) already produce:

| File | What it contains |
|---|---|
| `frames.csv` | Per-frame timestamps for each stage |
| `stage_energy.csv` | Energy attributed to each stage per run via INA3221 |
| `summary.json` | FPS stats, total energy, mean power |
| `sweep_summary.csv` | Aggregated across all (model × imgsz × precision × target_fps × repeat) |

The `stage_energy.csv` files are the key input for fitting the per-stage predictors.
We likely already have enough data to fit Stages 2–5. Stage 1 (capture) requires a
camera-only isolated benchmark (see Section 6.1).

---

## 6. Additional Benchmarks Needed

### Benchmark A — Capture only (no inference)

Run the camera capture + decode loop but skip the model entirely. This isolates
`E_capture` and `T_capture` as a function of resolution and FPS.

```bash
PYTHONPATH=src python -m camera_bench.cli \
    --bench-mode \
    --model none \          # skip model loading
    --width 640 --height 480 \
    --target-fps 0 \
    --duration-s 60 \
    --output-stream none
```

**Sweep:** `resolution ∈ {320×240, 640×480, 1280×720}` × `target_fps ∈ {0, 10, 20, 30}` × 3 repeats.

> If `--model none` is not yet supported, the simplest implementation is to add a
> `NullDetector` that returns empty detections instantly and has zero GPU usage.

### Benchmark B — Inference only (synthetic input, no camera)

Feed a pre-generated random tensor directly to the model in a loop, bypassing capture and
decode entirely. This cleanly isolates `E_infer + E_postprocess` from camera noise.

```python
# scripts/run_infer_only_bench.py
x = torch.randn(1, 3, imgsz, imgsz, device='cuda').to(dtype)
for _ in range(n_iterations):
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    record_timestamp()
```

**Sweep:** `model ∈ {yolov8n, ssdlite320, …}` × `imgsz ∈ {320, 640}` ×
`precision ∈ {fp16, fp32}` × 3 repeats (300 iterations each).

---

## 7. Sub-Predictor Design

### Stage 1 — Capture predictor

```
Inputs:  width, height, target_fps
Outputs: E_capture_per_frame (mJ), T_capture_per_frame (ms)
Data:    Benchmark A
Model:   Linear regression
         T_capture = max(T_hw_min, 1/target_fps)
         E_capture scales with pixel count and frame time
```

### Stage 2 — Preprocess predictor

```
Inputs:  width, height, imgsz, precision
Outputs: E_preprocess_per_frame (mJ), T_preprocess_per_frame (ms)
Data:    stage_energy.csv from existing sweeps
Model:   Linear regression (dominated by pixel count and memcpy size)
```

### Stage 3 — Inference predictor

```
Inputs:  model_flops, imgsz, precision
Outputs: E_infer_per_frame (mJ), T_infer_per_frame (ms)
Data:    stage_energy.csv from existing sweeps + Benchmark B
Model:   Gradient boosted trees (same as current predictor)
         This is a refinement of the existing model — now predicts T as well as E
```

### Stage 4 — Postprocess predictor

```
Inputs:  model_family (torchvision / yolo)
Outputs: E_post_per_frame (mJ), T_post_per_frame (ms)
Data:    stage_energy.csv from existing sweeps
Model:   Constant per model family (very small, low variance)
         YOLO: T_post = 0 (fused into Stage 3)
```

### Stage 5 — Overhead predictor

```
Inputs:  (none — treated as a system constant)
Outputs: P_overhead (W)  ← average background power draw
Data:    Estimated from idle periods in existing runs
Model:   Single constant value (~11–12 W based on observed low-FPS power floor)
```

---

## 8. Combination Layer

A single function that takes stage outputs and returns FPS + energy breakdown:

```python
def predict_pipeline(
    capture_pred,    # (E_mj, T_ms)
    preprocess_pred, # (E_mj, T_ms)
    infer_pred,      # (E_mj, T_ms)
    postprocess_pred,# (E_mj, T_ms)
    overhead_pred,   # P_w
    target_fps: float = 0,  # 0 = unbounded
) -> dict:

    T_compute_ms = (capture_pred.T + preprocess_pred.T +
                    infer_pred.T  + postprocess_pred.T)

    if target_fps > 0:
        T_frame_ms = max(T_compute_ms, 1000.0 / target_fps)
    else:
        T_frame_ms = T_compute_ms

    fps = 1000.0 / T_frame_ms

    E_overhead = overhead_pred.P * (T_frame_ms / 1000.0) * 1000  # mJ
    E_total    = (capture_pred.E + preprocess_pred.E +
                  infer_pred.E  + postprocess_pred.E + E_overhead)

    return {
        "fps":               fps,
        "E_total_mj":        E_total,
        "E_capture_mj":      capture_pred.E,
        "E_preprocess_mj":   preprocess_pred.E,
        "E_infer_mj":        infer_pred.E,
        "E_postprocess_mj":  postprocess_pred.E,
        "E_overhead_mj":     E_overhead,
        "bottleneck":        "throttle" if T_frame_ms > T_compute_ms else "compute",
    }
```

---

## 9. Validation Plan

| Test | Method | Target |
|---|---|---|
| **Per-stage accuracy** | Compare each sub-predictor against its held-out 20% test set | MAPE < 10% per stage |
| **End-to-end FPS** | Compare predicted FPS against measured FPS in existing sweep | Within ±1 fps |
| **End-to-end energy** | Compare predicted `E_total` against measured `energy_per_frame_j` | MAPE < 10% |
| **Low-FPS overhead** | Verify predicted energy rises correctly at target_fps = 5 | Correct trend |
| **New config** | Predict energy for a model/resolution not in training set; measure it | MAPE < 15% |

---

## 10. Implementation Checklist

- [ ] **Extract per-stage data** from existing `stage_energy.csv` files into a clean dataset
- [ ] **Benchmark A**: implement `--model none` / `NullDetector`, run capture-only sweep
- [ ] **Benchmark B**: write `scripts/run_infer_only_bench.py`, run inference-only sweep
- [ ] **Stage 1 predictor**: fit + validate (data from Benchmark A)
- [ ] **Stage 2 predictor**: fit + validate (data from existing sweeps)
- [ ] **Stage 3 predictor**: fit + validate (data from existing sweeps + Benchmark B)
- [ ] **Stage 4 predictor**: fit + validate (data from existing sweeps)
- [ ] **Stage 5 overhead**: estimate `P_overhead` from idle power floor
- [ ] **Combination layer**: implement `predict_pipeline()` in `src/energy_inference/`
- [ ] **End-to-end validation notebook**: per-stage plots + end-to-end accuracy
- [ ] **Generalisation test**: measure one unseen config, compare to prediction

---

## 11. Expected Outputs

1. **Five fitted sub-predictors** serialised as `.joblib` files
2. **`predict_pipeline()` function** in `src/energy_inference/predictor.py`
3. **Validation notebook** `benchmarks/decomposed_predictor_validation.ipynb`
4. **Per-stage energy breakdown plots** showing how each stage contributes across
   models and resolutions
