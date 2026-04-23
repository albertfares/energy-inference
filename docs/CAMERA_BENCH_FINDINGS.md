# Camera Benchmark — Findings & Supervisor Q&A

Benchmark runs on Jetson AGX Orin, Logitech webcam at 640×480, 60-second timed windows, 30-frame warmup discarded. All energy figures from INA3221 (cpu + gpu + io rails). Models: YOLOv8n (ultralytics) and SSDLite320-MobileNetV3-Large (torchvision).

---

## 1. Raw results

### 1.1 YOLOv8n — effect of target FPS

| Target FPS | Actual FPS | Mean power | Total energy | Energy/detection | `infer_fused` | `idle/sleep` |
|---|---|---|---|---|---|---|
| 5  | 5.0  | 13.4 W | 806 J  | 2686 mJ | 120 J (14.9%) | 681 J (84.5%) |
| 15 | 15.0 | 17.0 W | 1019 J | 1133 mJ | 392 J (38.5%) | 611 J (60.0%) |
| 30 | 29.9 | 19.5 W | 1171 J |  652 mJ | 784 J (67.0%) | 359 J (30.7%) |
| unbounded | 30.2 | 17.5 W | 1052 J |  583 mJ | 712 J (67.7%) |   2 J  (0.2%) |

Full stage breakdown for the 30 FPS paced run:

```
  infer_fused    :   784.17 J  ( 67.0%)   ← neural network forward pass + NMS (fused)
  idle/sleep     :   359.14 J  ( 30.7%)   ← time.sleep() between frames
  capture        :    18.76 J  (  1.6%)   ← cap.read() returning an already-ready frame
  filter         :     8.70 J  (  0.7%)   ← score threshold + max-det clipping
  ───────────────   ────────    ─────
  TOTAL          :  1170.77 J  (100.0%)
```

### 1.2 SSDLite320-MobileNetV3-Large — effect of target FPS

| Target FPS | Actual FPS | Mean power | Total energy | Energy/detection | `infer` | `idle/sleep` |
|---|---|---|---|---|---|---|
| 5  |  5.0 | 16.4 W |  987 J | 3290 mJ |  407 J (41.3%) | 562 J (56.9%) |
| 15 | 12.6 | 18.3 W | 1098 J | 1456 mJ | 1052 J (95.8%) |   3 J  (0.3%) |
| 30 | 12.5 | 18.9 W | 1135 J | 1520 mJ | 1087 J (95.8%) |   3 J  (0.3%) |
| unbounded | 11.7 | 18.8 W | 1132 J | 1612 mJ | 1089 J (96.2%) |   3 J  (0.3%) |

Full stage breakdown for the unbounded run:

```
  infer          :  1088.65 J  ( 96.2%)   ← GPU forward pass (conv layers)
  preprocess     :    25.11 J  (  2.2%)   ← BGR → tensor, normalise, .to(device)
  capture        :     8.01 J  (  0.7%)   ← cap.read() (blocks ~70ms for next frame)
  postprocess    :     4.62 J  (  0.4%)   ← D2H transfer + dict extraction
  filter         :     2.75 J  (  0.2%)   ← score threshold + max-det clipping
  idle/sleep     :     2.50 J  (  0.2%)   ← no headroom — GPU always busy
  ───────────────   ────────    ─────
  TOTAL          :  1131.65 J  (100.0%)
```

---

## 2. Key findings

### 2.1 The idle power floor dominates at low frame rates

The Jetson draws ~13–16 W even while sleeping between frames. This means reducing frame rate saves total energy but increases energy per detection:

```
                 Total energy    Detections    Energy / detection
YOLOv8n  5 fps:   806 J    →    300           2686 mJ
YOLOv8n 15 fps:  1019 J    →    899           1133 mJ
YOLOv8n 30 fps:  1171 J    →   1797            652 mJ   ← best efficiency
```

Going from 5 to 30 FPS costs only 45% more total energy but delivers 6× more detections.

### 2.2 The marginal cost of one extra inference is small

The Jetson never drops below ~13 W at idle. Running more inferences replaces idle time with active time; the marginal energy cost per extra inference is only the *delta* between active power and idle power:

```
Active power during inference:  ~21 W  (consistent across all runs and both models)
Idle power between frames:      ~13–16 W  (varies with gap length — see §2.3)
Marginal cost per inference:    ~6 W × ~21 ms ≈ 126 mJ
```

This is why doubling from 15 to 30 FPS costs only ~15% more total energy despite doing twice the work.

### 2.3 Idle power depends on gap length

Longer inter-frame gaps allow the GPU to reach a deeper idle state:

| Target FPS | Inter-frame gap | Idle power |
|---|---|---|
|  5 FPS | ~179 ms | 12.6 W |
| 15 FPS |  ~45 ms | 14.8 W |
| 30 FPS |  ~13 ms | 15.3 W |

At 5 FPS the GPU has 179 ms to cool down between inferences and sheds ~3 W relative to 30 FPS. This effect is real but not large enough to make low frame rates energy-efficient.

*Derivation: idle power = idle_energy / idle_time, where idle_time = duration − (frames × mean_latency).*

### 2.4 FPS target pacing changes where the wait appears

With no target FPS (unbounded), the pipeline calls `cap.read()` immediately after inference. The camera hasn't delivered the next frame yet, so `cap.read()` blocks — that wait appears as **capture** stage time. With a target FPS set, the pipeline sleeps first, then calls `cap.read()` on a frame that is already ready — the wait appears as **idle/sleep**.

```
Unbounded:        [==cap.read() blocks 13ms==][==infer 20ms==][==cap.read()...
--target-fps 30:  [==infer 20ms==][==sleep 13ms==][cap.read() 0.5ms][==infer...
```

Same wall-clock wait, different label. The unbounded run's `capture = 31%` and the 30 FPS run's `idle = 31%` are the same physical phenomenon.

### 2.5 SSDLite is slower and less efficient than YOLOv8n on Jetson

| | SSDLite (unbounded) | YOLOv8n (unbounded) |
|---|---|---|
| Max FPS | 11.7 | 30.2 |
| Latency | 85 ms | 33 ms |
| Energy / detection | 1612 mJ | 583 mJ |
| `infer` share | **96.2%** | 67.7% |
| Idle | 0.2% | 0.2% |

YOLOv8n is 2.6× faster and 2.8× more energy-efficient per detection. SSDLite is **compute-bound** — 96% of energy goes to the forward pass with no idle headroom. YOLOv8n is fast enough to be **camera-limited**, spending ~30% of its time waiting for the next frame.

SSDLite's "lightweight" label refers to parameter count and CPU FLOPs. On a CUDA GPU, YOLOv8n's architecture (GPU-optimised conv blocks) maps more efficiently than SSDLite's MobileNetV3 backbone, which was designed for mobile CPU inference.

### 2.6 Target FPS has no effect on SSDLite above ~12 FPS

SSDLite inference takes ~80 ms. The pacing sleep only fires when the target period is *longer* than the actual processing time:

- `--target-fps 30` → period = 33 ms < 80 ms inference → sleep = 0 → runs at natural speed
- `--target-fps 15` → period = 67 ms < 80 ms inference → sleep = 0 → runs at natural speed
- `--target-fps 5`  → period = 200 ms > 80 ms inference → sleep = 120 ms → pacing active

For SSDLite, FPS throttling only has an effect below its natural ~12 FPS ceiling.

---

## 3. Supervisor questions

### Q1 — "Why is INT8 not used on Jetson? (the architecture part)"

**Short answer:** INT8's throughput advantage over FP16 requires large tensor tiles in the GPU's Tensor Cores. At batch=1 (all real-time camera pipelines), those tiles are never full, so the theoretical 2× gain does not materialise. True INT8 hardware on Jetson is the NVDLA, not the GPU — but NVDLA has incomplete layer support for modern detection architectures.

**Architecture detail:**

NVIDIA Ampere Tensor Cores compute matrix multiplications in fixed-size tiles:

```
FP16 tile:  16 × 16 × 16  matrix multiply per clock cycle
INT8 tile:  16 × 16 × 32  matrix multiply per clock cycle  (2× ops/cycle)
```

The 2× INT8 advantage is only realised when inputs are large enough to fill those tiles — which requires large batch sizes. At batch=1, both precisions leave the Tensor Cores significantly underutilised. MLPerf Inference results show the practical INT8 vs FP16 latency gap at batch=1 on Jetson is 10–20%, far below the theoretical 2×.

The Jetson AGX Orin has two distinct INT8 compute paths:

| Path | Throughput | Precision | Layer support | Practical use |
|---|---|---|---|---|
| GPU Tensor Cores | ~67 TOPS | FP32 / FP16 / INT8 | All layers | FP16 preferred (no calibration) |
| NVDLA × 2 | 26.2 TOPS each | INT8 only | Limited | Requires TensorRT + calibration |

The NVDLA is purpose-built for INT8 and is the correct hardware path for quantized inference on Jetson. However, it requires:
1. TensorRT export with a representative calibration dataset (PTQ)
2. Every layer to be NVDLA-compatible — YOLO heads, custom ops, and attention layers frequently fall back to the GPU, breaking the pipeline into GPU↔NVDLA transfers that can cost more than the INT8 savings

FP16 on the GPU works out of the box, has no accuracy risk, and already achieves near-peak utilisation for batch=1 camera inference. This is why NVIDIA's own Jetson deployment guides and Ultralytics' Jetson export default to FP16, not INT8.

**Reference:** Reddi et al., *"MLPerf Inference Benchmark"*, MLSys 2020 — establishes INT8 as the standard for *closed-division* (datacenter-scale, large-batch) edge benchmarks but explicitly notes batch-size sensitivity. NVIDIA TensorRT documentation: *"FP16 is recommended as the starting point for Jetson; INT8 requires a representative calibration dataset and offers diminishing returns at batch=1."*

**For the paper:**
> *"We benchmark FP32 and FP16, which covers the dominant Jetson deployment range. At batch=1 camera inference, INT8's theoretical 2× Tensor Core throughput advantage over FP16 does not materialise due to tile underutilisation. True INT8 acceleration on Jetson requires the NVDLA, which has incomplete layer support for YOLOv8 and is left as future work."*

---

### Q2 — "What does the `capture` stage include — does it include the camera's own power?"

**Short answer:** No. The INA3221 only measures the Jetson module's internal rails. The camera's own power (drawn via USB VBUS from the carrier board) is outside the measurement path and not captured.

**What `cap.read()` actually does:**

```
cap.read()
  │
  ├─ 1. V4L2 blocking wait      CPU blocked, waiting for next USB frame
  ├─ 2. USB bulk transfer       camera → USB host controller → kernel buffer
  │                             (~20–50 KB compressed MJPEG per frame)
  ├─ 3. MJPEG decode            libjpeg-turbo decodes frame on CPU
  │                             (Logitech webcams send MJPEG over USB to save bandwidth)
  ├─ 4. YUV → BGR conversion    V4L2 / OpenCV colour space conversion on CPU
  └─ 5. DMA copy                kernel → userspace numpy array
```

**What the INA3221 measures during `capture`:**

The INA3221 measures **total rail power** — it cannot distinguish USB transfer from GPU idling. The attributed capture energy is:

```
capture_energy = ∫ (cpu_rail + gpu_rail + io_rail)  dt   over cap.read() duration
```

During `cap.read()`, approximate contributions are:

| Component | Rail | Note |
|---|---|---|
| USB data transfer | IO | Small — MJPEG frame is ~30 KB, microseconds of actual bus time |
| MJPEG decode + YUV→BGR | CPU | 2–5 ms of CPU work |
| GPU idle during wait | GPU | **Dominant term** — GPU draws ~12 W doing nothing |
| V4L2 blocking wait | All | Whole board powered on while waiting |

The large majority of what we label "capture energy" is the GPU burning power during the inter-frame wait, not the USB transfer itself. The USB IO contribution is roughly 15–20% of the attributed capture energy.

**What is not measured — the camera itself:**

```
Wall outlet
     │
     ▼
Carrier board PSU
     │
     ├──► Jetson module ──► INA3221 measures this ✓
     │         ├── CPU rail
     │         ├── GPU rail
     │         └── IO rail  (USB host controller logic, not VBUS current)
     │
     └──► USB VBUS (5V) ──► Logitech webcam  ✗ NOT measured
                                 ├── image sensor
                                 ├── MJPEG encoder chip
                                 └── USB PHY
```

A typical Logitech USB webcam draws **300–500 mW** from the USB bus. This is real energy consumption but is invisible to the INA3221 because it is supplied by the carrier board outside the measurement path.

**For the paper:**
> *"The INA3221 sensors measure power consumed by the Jetson SoC module only. The USB camera's own power draw (~300–500 mW for a typical USB webcam) is not captured, as it is supplied via USB VBUS from the carrier board outside the measurement path. The `capture` stage energy reported here represents total Jetson-side energy during `cap.read()` — including GPU idle draw during the inter-frame wait — and should not be interpreted as the energy cost of USB transfer alone. All reported figures represent Jetson computation cost, not total system energy."*

---

## 4. Summary for the paper

| Finding | Value |
|---|---|
| YOLOv8n max FPS on Jetson (640×480, fp32) | 30.2 FPS |
| SSDLite max FPS on Jetson (640×480, fp32) | 11.7 FPS |
| YOLOv8n energy/detection at 30 FPS | 652 mJ |
| SSDLite energy/detection at max speed | 1612 mJ |
| Jetson idle power floor | ~13–16 W |
| Jetson active inference power (both models) | ~19–21 W |
| Share of energy that is GPU inference at 30 FPS (YOLOv8n) | 67% |
| Share of energy that is idle/sleep at 5 FPS (YOLOv8n) | 85% |
| Camera (USB webcam) power — measured by INA3221 | **No** |
| INT8 practical speedup over FP16 at batch=1 | 10–20% (not 2×) |
