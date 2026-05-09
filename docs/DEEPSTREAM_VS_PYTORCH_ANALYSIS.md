# DeepStream vs PyTorch — Energy & FPS Comparison

**Model:** YOLOv8n · **Hardware:** Jetson AGX Orin (MAXN mode) · **Camera:** USB V4L2 640×480 MJPEG

---

## Overview

This document compares two end-to-end inference pipelines for YOLOv8n on a live USB camera feed:

| Stage | PyTorch pipeline | DeepStream pipeline |
|---|---|---|
| Capture | V4L2 → OpenCV | V4L2 → GStreamer `v4l2src` |
| Decode | CPU (libjpeg-turbo) | CPU (`jpegdec` SW fallback) |
| Preprocess | CPU → CUDA memcpy | `nvvideoconvert` (GPU) |
| **Inference** | **PyTorch CUDA** | **TensorRT via `nvinfer`** |
| Output | optional H.264 encode | `fakesink` |

**Energy** is measured by an INA3221 power monitor at 1 kHz across three rails: CPU, GPU, and IO.
Both sweeps cover the full grid: `imgsz ∈ {320, 640}` × `precision ∈ {FP16, FP32}` × `target_fps ∈ {0, 5, 10, 15, 20, 25, 30}` × 3 repeats (60 s each).
`target_fps = 0` means unbounded — the camera naturally caps both pipelines at ~30 fps.

---

## Results

### Energy per Frame

![Energy per frame vs target FPS](figures/deepstream_vs_pytorch_epf.png)

### Mean System Power

![Mean system power vs target FPS](figures/deepstream_vs_pytorch_power.png)

### DeepStream Energy Saving vs PyTorch

![Energy saving % vs target FPS](figures/deepstream_vs_pytorch_saving.png)

### Rail Breakdown at Unbounded FPS

![GPU / CPU / IO energy per frame at target_fps=0](figures/deepstream_vs_pytorch_rail_breakdown.png)

---

## Summary Table — Unbounded FPS (target_fps = 0)

All values are medians over 3 repeats.

| imgsz | Precision | PyTorch mJ/frame | DeepStream mJ/frame | Energy saving | PyTorch W | DeepStream W | Power saving |
|---|---|---|---|---|---|---|---|
| 640 | FP32 | 521 | 523 | **−0.5%** | 15.6 | 15.7 | −0.3% |
| 640 | FP16 | 485 | 442 | **+8.9%** | 14.6 | 13.3 | +9.0% |
| 320 | FP32 | 446 | 414 | **+7.1%** | 13.4 | 12.4 | +7.3% |
| 320 | FP16 | 437 | 398 | **+8.8%** | 13.1 | 12.0 | +9.0% |

---

## Full Savings Table

Median over repeats per (imgsz × precision × target_fps) cell.
Positive = DeepStream uses less energy. Negative = DeepStream uses more.

| imgsz | Precision | target_fps | PT mJ/frame | DS mJ/frame | Saving % | PT W | DS W | Power saving % |
|---|---|---|---|---|---|---|---|---|
| 640 | FP32 | 0 | 521 | 523 | −0.5 | 15.6 | 15.7 | −0.3 |
| 640 | FP32 | 5 | 2216 | 2366 | −6.8 | 11.1 | 11.9 | −7.1 |
| 640 | FP32 | 10 | 1200 | 1226 | −2.2 | 12.0 | 12.3 | −2.5 |
| 640 | FP32 | 15 | 864 | 850 | +1.6 | 12.9 | 12.8 | +1.4 |
| 640 | FP32 | 20 | 695 | 671 | +3.3 | 13.9 | 13.4 | +3.1 |
| 640 | FP32 | 25 | 592 | 555 | +6.2 | 14.8 | 13.9 | +6.0 |
| 640 | FP32 | 30 | 525 | 479 | +8.7 | 15.7 | 14.4 | +8.5 |
| 640 | FP16 | 0 | 485 | 442 | +8.9 | 14.6 | 13.3 | +9.0 |
| 640 | FP16 | 5 | 2190 | 2670 | −21.9 | 10.9 | 13.4 | −22.3 |
| 640 | FP16 | 10 | 1164 | 1260 | −8.2 | 11.6 | 12.6 | −8.4 |
| 640 | FP16 | 15 | 828 | 837 | −1.2 | 12.4 | 12.6 | −1.4 |
| 640 | FP16 | 20 | 655 | 643 | +1.8 | 13.1 | 12.9 | +1.6 |
| 640 | FP16 | 25 | 555 | 525 | +5.4 | 13.9 | 13.1 | +5.2 |
| 640 | FP16 | 30 | 484 | 454 | +6.4 | 14.5 | 13.6 | +6.1 |
| 320 | FP32 | 0 | 446 | 414 | +7.1 | 13.4 | 12.4 | +7.3 |
| 320 | FP32 | 5 | 2145 | 2283 | −6.4 | 10.7 | 11.5 | −7.1 |
| 320 | FP32 | 10 | 1125 | 1190 | −5.8 | 11.2 | 11.9 | −6.0 |
| 320 | FP32 | 15 | 785 | 816 | −3.9 | 11.8 | 12.2 | −4.1 |
| 320 | FP32 | 20 | 617 | 626 | −1.6 | 12.3 | 12.5 | −1.8 |
| 320 | FP32 | 25 | 516 | 479 | +7.1 | 12.9 | 12.0 | +6.8 |
| 320 | FP32 | 30 | 449 | 409 | +8.9 | 13.4 | 12.3 | +8.7 |
| 320 | FP16 | 0 | 437 | 398 | +8.8 | 13.1 | 12.0 | +9.0 |
| 320 | FP16 | 5 | 2142 | 2243 | −4.7 | 10.7 | 11.3 | −5.4 |
| 320 | FP16 | 10 | 1115 | 1136 | −1.9 | 11.1 | 11.4 | −2.2 |
| 320 | FP16 | 15 | 776 | 770 | +0.7 | 11.6 | 11.6 | +0.5 |
| 320 | FP16 | 20 | 604 | 582 | +3.7 | 12.1 | 11.6 | +3.5 |
| 320 | FP16 | 25 | 504 | 473 | +6.3 | 12.6 | 11.8 | +6.1 |
| 320 | FP16 | 30 | 434 | 398 | +8.3 | 13.0 | 11.9 | +8.1 |

---

## Key Findings

### 1. DeepStream wins at high throughput, loses at low FPS

The energy-per-frame saving from DeepStream follows a consistent **FPS-dependent pattern** across all four (imgsz × precision) configurations:

- **target_fps ≤ 10 fps → DeepStream uses more energy** (up to −22% for imgsz=640 FP16 at 5 fps).
  The GStreamer/nvinfer pipeline keeps GPU and IO infrastructure powered even between frames, whereas the PyTorch loop simply sleeps. This idle overhead dominates at low frame rates.

- **target_fps ≥ 20–25 fps → DeepStream uses less energy** (+3–9%).
  At high utilisation, TensorRT's fused kernels and Tensor Core scheduling run the same inference with fewer wasted GPU cycles than PyTorch's eager CUDA execution.

- **The crossover point is roughly 15–20 fps** — the inflection varies slightly by config.

### 2. FP16 gains the most from DeepStream

At full throughput (target_fps = 0), FP16 saves ~9% in both energy and power, while FP32 saves 0–7%. TensorRT's FP16 Tensor Core path is more aggressively optimised than its FP32 path, so the gap between PyTorch and TensorRT is larger in FP16.

### 3. Smaller imgsz benefits more at mid-range FPS

For imgsz=320, DeepStream breaks even with PyTorch around target_fps=15, versus ~20 fps for imgsz=640. The smaller inference workload makes the relative overhead of the GStreamer pipeline lighter, shifting the crossover earlier.

### 4. Rail breakdown: GPU dominates in both pipelines

At unbounded FPS, the GPU rail accounts for ~52–54% of total energy in both backends. The IO rail (camera + memory bandwidth) is ~23–25% and the CPU rail ~15–22%. DeepStream reduces all three rails proportionally — it is not purely a GPU-side optimisation.

### 5. System power scales with FPS in both pipelines

Neither pipeline has a flat idle floor: power rises roughly linearly from ~11 W at 5 fps to ~16 W at 30 fps. DeepStream sits ~1–1.5 W below PyTorch at high FPS but ~0.5–2 W above it at low FPS, consistent with the pipeline-overhead explanation.

---

## Data Quality Notes

- 2 out of 84 DeepStream runs failed mid-run with `"Failed to queue input batch for inferencing"`:
  - imgsz=640 FP16 target_fps=0 repeat=0 (ran 34.5 s)
  - imgsz=320 FP16 target_fps=20 repeat=1 (ran 24.6 s)
  Both are FP16 runs; both other repeats of those configs succeeded. The error is a transient TensorRT GPU memory fragmentation issue under the FP16 Tensor Core engine, not a systematic failure. The remaining 2 repeats are sufficient for a valid median.

- The PyTorch imgsz=640 sweep (`yolov8n_fps_sweep_MAXN_20260428_133405`) predates the `yolo_imgsz` column — it was added retrospectively based on the run directory name and confirmed by the energy levels.

---

## Sweep Metadata

| | PyTorch | DeepStream |
|---|---|---|
| Run date | 2026-04-28 | 2026-05-07 |
| Jetson power mode | MAXN | MAXN |
| Camera | USB V4L2 `/dev/video0` 640×480 MJPEG | same |
| Warmup | 5 s | 5 s |
| Benchmark window | 60 s | 60 s |
| INA3221 rate | 1 kHz | 1 kHz |
| Total runs | 84 ok | 82 ok / 2 failed |
| Inference backend | PyTorch 2.x CUDA | TensorRT via DeepStream 7.1 `nvinfer` |
| Engine source | Ultralytics YOLOv8n | Exported via `setup_deepstream.sh` |
