# Decomposed Energy Predictor — Design Plan

**Status:** Planning  
**Hardware:** Jetson AGX Orin (MAXN)  
**Backends covered:** PyTorch CUDA, DeepStream/TensorRT  

---

## 1. Motivation

The current predictor takes `(model, imgsz, precision, target_fps)` as inputs and outputs
`energy_per_frame`. This design has two problems:

1. **FPS is an input, not an output.** In a real deployment you don't choose FPS — it emerges
   from the hardware. To predict energy at a new operating point you have to guess the FPS first.

2. **The pipeline is treated as a black box.** There is no way to reason about where the energy
   goes, or to swap one component (e.g. a different camera resolution) without re-running the
   full sweep.

The new design decomposes the pipeline into independent stages. Each stage has its own
sub-predictor that only depends on the parameters relevant to that stage. FPS and total energy
are then **derived** by combining the stage outputs.

---

## 2. Pipeline Decomposition

The end-to-end inference pipeline breaks down as follows:

```
[Camera]  →  [Decode + Preprocess]  →  [Inference]  →  [Output / Overhead]
```

For each stage we want to predict two quantities:
- **`E_stage` (J/frame)** — energy consumed by this stage per processed frame
- **`T_stage` (s/frame)** — time taken by this stage per frame (its latency contribution)

### Stage 1 — Camera Capture

| | |
|---|---|
| **What it covers** | V4L2 source, MJPEG capture, DMA transfer to host |
| **Key parameters** | Resolution (`W × H`), format (MJPEG), capture framerate cap |
| **Independent of** | Model, precision, inference backend |
| **Notes** | At low target FPS the camera idles between captures; `T_camera` includes that idle time. At unbounded FPS, `T_camera = 1 / camera_max_fps`. |

### Stage 2 — Decode + Preprocess

| | |
|---|---|
| **What it covers** | JPEG decode (`jpegdec` SW or `nvjpegdec` HW) + `nvvideoconvert` / CPU resize + CUDA memcpy |
| **Key parameters** | Resolution, decode backend (SW / HW), output format (NV12 / RGB) |
| **Independent of** | Model, precision |
| **Notes** | Currently SW decode is used for both backends (hardware JPEG decode unstable). If HW decode becomes available it changes both `E_decode` and `T_decode`. |

### Stage 3 — Inference

| | |
|---|---|
| **What it covers** | Forward pass: PyTorch CUDA kernel or TensorRT `nvinfer` |
| **Key parameters** | Model (`yolov8n`, `ssdlite320`, …), `imgsz`, `precision` (FP16/FP32), backend (PyTorch / TRT) |
| **Independent of** | Camera resolution, decode backend |
| **Notes** | This is the core ML component and the primary target of the TRT optimisation study. |

### Stage 4 — Pipeline Overhead

| | |
|---|---|
| **What it covers** | GStreamer/DeepStream infrastructure, Python event loop, memory allocation, idle GPU kernel scheduling |
| **Key parameters** | Backend (DeepStream has ~1–2 W higher idle draw than PyTorch) |
| **Independent of** | Model, resolution (approximately) |
| **Notes** | Roughly constant per frame at high FPS; becomes the dominant term at low FPS. |

---

## 3. How FPS and Total Energy Are Derived

### 3.1 Time model

In a pipelined system, stages run in parallel. The throughput is limited by the slowest stage
(the bottleneck):

```
T_frame = max(T_camera, T_decode + T_infer)
FPS     = 1 / T_frame
```

- If `T_camera > T_decode + T_infer` → **camera-limited** (e.g. YOLOv8n at 30 fps)
- If `T_decode + T_infer > T_camera` → **compute-limited** (e.g. SSDLite at ~15 fps)

> **Why "sum up times":** the supervisor's suggestion to sum times refers to the compute branch
> (`T_decode + T_infer`). The camera runs concurrently; whichever branch takes longer sets the
> frame rate. FPS is then `n_frames / T_total_run = 1 / max(T_camera, T_compute)`.

### 3.2 Energy model

Energy is always a sum — all stages draw power simultaneously regardless of which one is the
bottleneck:

```
E_total_per_frame = E_camera + E_decode + E_infer + E_overhead
```

Note that `E_overhead` is measured per unit time, not per frame, so it grows when FPS is low:

```
E_overhead_per_frame = P_overhead × T_frame
```

This naturally explains the observed pattern: **at low FPS, DeepStream costs more energy per
frame than PyTorch**, because its higher idle power (`P_overhead`) is multiplied by a longer
`T_frame`.

---

## 4. Data Collection Plan

### 4.1 What we already have

The existing FPS sweeps (PyTorch and DeepStream, both models) already provide:
- Total system energy split by INA3221 rail (CPU / GPU / IO)
- Actual FPS achieved at each target FPS
- Enough coverage to fit Stage 3 (inference) and Stage 4 (overhead) predictors

**Rail-to-stage mapping (approximate):**

| INA3221 rail | Primary contributor |
|---|---|
| GPU rail | Stage 3 (inference) |
| CPU rail | Stage 2 (decode + preprocess) + Stage 4 overhead |
| IO rail | Stage 1 (camera) + memory bandwidth |

### 4.2 Isolated benchmarks needed

To get clean per-stage measurements, three additional targeted experiments are needed:

#### Benchmark A — Camera + Decode only (no inference)

Run the full capture + decode + preprocess chain but replace `nvinfer` with `fakesink` directly
(no model loaded). Measures `E_camera + E_decode` in isolation.

```bash
# Proposed: add a --no-infer flag to run_deepstream_bench.py
python3 scripts/run_deepstream_bench.py \
    --model none --no-infer \
    --width 640 --height 480 \
    --target-fps 0 --duration 60
```

Sweep: `resolution ∈ {320×240, 640×480, 1280×720}` × `target_fps ∈ {0, 10, 20, 30}` × 3 repeats.

#### Benchmark B — Inference only (synthetic input)

Feed a pre-generated random tensor directly to the model, bypassing camera and decode entirely.
This isolates `E_infer + E_overhead`.

For PyTorch:
```python
# Loop: generate random tensor, run model forward pass, record energy
x = torch.randn(1, 3, imgsz, imgsz, device='cuda')
with torch.no_grad():
    _ = model(x)
```

For DeepStream/TRT: use `appsrc` instead of `v4l2src` to inject synthetic NV12 frames.

Sweep: `model ∈ {yolov8n, ssdlite320}` × `imgsz ∈ {320, 640}` × `precision ∈ {fp16, fp32}` ×
`backend ∈ {pytorch, trt}` × 3 repeats.

#### Benchmark C — Overhead floor (idle pipeline)

Start the full pipeline (camera + decode + preprocess + nvinfer loaded) but use
`interval=999` in nvinfer (run inference once every 1000 frames — effectively idle).
This isolates `E_overhead` — the fixed infrastructure cost.

---

## 5. Sub-Predictor Design

Each sub-predictor is a small regression model fit on the isolated benchmark data.

### Stage 1 — Camera predictor

```
Inputs:  width, height, target_fps
Outputs: E_camera_per_frame (mJ), T_camera_per_frame (ms)
Model:   Linear or polynomial regression
         (camera energy scales roughly with pixel count and frame rate)
```

### Stage 2 — Decode predictor

```
Inputs:  width, height, decode_backend (SW=0 / HW=1)
Outputs: E_decode_per_frame (mJ), T_decode_per_frame (ms)
Model:   Linear regression
         (SW decode scales with pixel count; HW decode roughly constant)
```

### Stage 3 — Inference predictor

```
Inputs:  model_flops, imgsz, precision (FP16=0 / FP32=1), backend (PT=0 / TRT=1)
Outputs: E_infer_per_frame (mJ), T_infer_per_frame (ms)
Model:   Gradient boosted trees (same approach as current predictor)
         Trained on both existing sweep data and Benchmark B data
```

### Stage 4 — Overhead predictor

```
Inputs:  backend (pytorch / deepstream)
Outputs: P_overhead (W)  ← power draw, not energy per frame
         (energy per frame = P_overhead × T_frame, computed at combination time)
Model:   Constant per backend (estimated from Benchmark C)
```

---

## 6. Combination Layer

A thin Python function that takes the four sub-predictor outputs and returns the final predictions:

```python
def predict_pipeline(
    camera_pred,   # (E_camera, T_camera)
    decode_pred,   # (E_decode, T_decode)
    infer_pred,    # (E_infer,  T_infer)
    overhead_pred, # P_overhead
) -> dict:

    T_compute = decode_pred.T + infer_pred.T   # serial compute branch
    T_frame   = max(camera_pred.T, T_compute)  # bottleneck
    fps       = 1.0 / T_frame

    E_overhead_per_frame = overhead_pred.P * T_frame
    E_total  = (camera_pred.E + decode_pred.E +
                infer_pred.E + E_overhead_per_frame)

    return {
        "fps":                fps,
        "E_total_mj":         E_total,
        "E_camera_mj":        camera_pred.E,
        "E_decode_mj":        decode_pred.E,
        "E_infer_mj":         infer_pred.E,
        "E_overhead_mj":      E_overhead_per_frame,
        "bottleneck":         "camera" if camera_pred.T >= T_compute else "compute",
    }
```

---

## 7. Validation Plan

| Test | Method |
|---|---|
| **Per-stage accuracy** | Compare each sub-predictor's output against its isolated benchmark (held-out 20%) |
| **End-to-end accuracy** | Compare combined prediction against full pipeline sweep (existing data) |
| **Generalisation** | Predict energy for a new (model, resolution, backend) combination not in the training set; verify against a new measurement |
| **Bottleneck identification** | Confirm the model correctly predicts camera-limited vs compute-limited regimes for YOLOv8n (camera) and SSDLite (compute) |

Target: end-to-end MAPE < 10% across all tested configurations.

---

## 8. Implementation Checklist

- [ ] **Benchmark A**: add `--no-infer` mode to `run_deepstream_bench.py`, run sweep
- [ ] **Benchmark B**: write `scripts/run_infer_only_bench.py` (PyTorch + TRT appsrc), run sweep
- [ ] **Benchmark C**: run idle pipeline sweep (nvinfer `interval=999`)
- [ ] **Stage 1 predictor**: fit + validate camera predictor
- [ ] **Stage 2 predictor**: fit + validate decode predictor
- [ ] **Stage 3 predictor**: fit + validate inference predictor (reuse + extend existing work)
- [ ] **Stage 4 predictor**: estimate overhead constant per backend
- [ ] **Combination layer**: implement `predict_pipeline()` function
- [ ] **End-to-end validation**: compare against existing sweep data
- [ ] **New-config generalisation test**: measure one unseen config, compare to prediction

---

## 9. Expected Outputs

1. **Four fitted sub-predictors** (serialised as `.joblib` or `.pkl`)
2. **`predict_pipeline()` function** in `src/energy_inference/`
3. **Validation notebook** with per-stage and end-to-end accuracy plots
4. **Updated analysis document** interpreting results in terms of stage contributions
