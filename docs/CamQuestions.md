# Camera Use Case — Usage Study Summary

> Brief findings from literature/industry sources on (1) how detection cameras are typically deployed, and (2) how vision model performance is evaluated.

---

## 1. How are cameras typically run for detection?

### Frame rate (FPS)
The deployed FPS is much lower than camera capture rates suggest. An IPVM survey of 80+ integrators reports **~70% record at ≤10 FPS**, with an industry average of **6–10 FPS**.

| FPS range | Typical use |
|---|---|
| **5–10** | Large-area / low-priority surveillance, storage-constrained deployments |
| **10–15** | Standard surveillance sweet spot — captures a walking person reliably |
| **15–30** | High-value zones: entrances, retail, traffic, parking lots |
| **30+** | Specialized only (casinos, sports, fast motion) |

**Key principle:** required FPS is set by the speed of the objects of interest, not by hardware capacity.
- Traffic/fast objects → ≥10 FPS
- People counting / slow scenes → 2–3 FPS sufficient

Embedded edge deployments commonly target ~10 FPS (e.g., pruned YOLOv4 at 416×256 on an embedded camera achieved ~11.8 FPS, deemed acceptable for intersection monitoring).

### Resolution
- **Capture resolution** (camera): 1080p / 4K (for forensics, zoom).
- **Detector input resolution**: much lower — common YOLO defaults are 320, 416, 640.
- Trade-off: higher input → better small-object detection range, but higher latency/energy.

### Precision
- **PyTorch / research**: FP32, FP16, BF16.
- **Production edge**: **INT8 via TensorRT is the deployment standard.**

### Batch size
- For real-time camera processing: **batch=1 is universal.** Frames arrive one at a time; queueing adds unacceptable latency.
- batch > 1 only appears in (a) multi-camera systems (one accelerator, N streams) or (b) offline/recorded video processing.

### Common architectural patterns
1. **Always-on full-rate detection** — every frame, fixed FPS. Simple, energy-expensive.
2. **Adaptive / motion-triggered** — low FPS idle, high FPS on motion. Standard in modern NVR systems (e.g., Frigate). Large energy savings.
3. **Detect-every-N + lightweight tracking** — heavy detector every N frames + tracker (Kalman, SORT, ByteTrack) in between.

---

## 2. How is vision model performance evaluated?

Two independent axes are always reported together.

### Axis A — Accuracy
- **IoU (Intersection over Union)** — overlap between predicted and ground-truth box. Threshold typically 0.5.
- **Precision** / **Recall** / **F1**.
- **AP (Average Precision)** — area under precision-recall curve, per class.
- **mAP (mean Average Precision)** — AP averaged across classes. *The* standard accuracy metric.
  - **mAP@0.5** (PASCAL VOC style) — single threshold, lenient.
  - **mAP@0.5:0.95** (COCO style) — averaged over 10 IoU thresholds, stricter, modern default.

### Axis B — Speed / Efficiency
- **Latency** (ms/frame).
- **Throughput / FPS** (inferences per second).
- **Model size** (parameters, MB).
- **Memory usage** (peak GPU RAM).
- **Energy per inference (J/frame)** — increasingly reported in edge papers, not yet standard.

### "Accuracy per time period" — practical framings
There is no single canonical metric. Common approaches:

1. **Accuracy–FPS Pareto curve** — the dominant approach. Plot mAP vs FPS, one point per model/config. Every YOLO release publishes this.
2. **mAP at a fixed FPS budget** — e.g., "best mAP achievable at 30 FPS on this hardware."
3. **Video-specific:**
   - Frame-level mAP (per-frame, expensive).
   - Tracking metrics (MOTA, HOTA, IDF1) — account for identity persistence and ID switches.
   - Effective recall over a time window — "did we detect the object at least once during its appearance?" Operationally relevant for surveillance, rarely formalized.
4. **Energy-aware metrics** (project-specific):
   - **mAP per Joule.**
   - **Joules per true detection.**

---

## 3. Implications for the project

- **Make batch=1 the headline configuration** for the camera use case. batch=4 results, while measured, are not deployment-realistic for real-time camera pipelines.
- **Anchor benchmarks to operational FPS targets** (e.g., 5 / 15 / 30 FPS) rather than reporting raw latency alone — this matches how the field talks about deployment.
- **The natural framing for the predictor:** *given an FPS target on the Jetson, which model/config achieves the best mAP at the lowest energy?* This is a 3D Pareto (mAP × FPS × Joules) — directly enabled by an energy predictor.
- **For the camera benchmark, measure the full pipeline** (capture → decode → resize → normalize → inference → post-process), not just `model(tensor)`. Preprocessing energy is non-trivial and is invisible in the current synthetic-tensor sweep.
- **Plan a sustained run** (5–10 min on the webcam) to check thermal stability and extrapolate to 24/7 operation — the relevant deployment scenario.
- **Open question — accuracy measurement**: the Logitech webcam alone provides no ground truth. Three options:
  1. Inference-only benchmarking (latency + energy, no mAP).
  2. Use a labeled video dataset (MOT17, VisDrone, BDD100K) for the joint accuracy/energy analysis; use the webcam only for end-to-end pipeline validation.
  3. Manually annotate a short webcam clip as a small mAP probe.

  Option 2 is the most defensible for a publishable result.

---

## 4. Camera capture format (USB webcam)

USB webcams typically expose multiple pixel formats. Our Logitech offers `YUYV` (uncompressed), `MJPG` (JPEG per frame), and `H264` (full video compression on-camera). Trade-off:

| Format | Compression | Per-frame CPU on Jetson | Max resolution at 30 FPS |
|---|---|---|---|
| `YUYV` | none | very low (just YUV→BGR) | 640×480 |
| `MJPG` | JPEG per frame | low (libjpeg-turbo decode) | 1920×1080 |
| `H264` | full inter-frame | requires hardware decode pipeline (complex) | 1920×1080 |

**Decision for the benchmark:** the format is exposed as a CLI axis `--capture-format {yuyv, mjpg, auto}`, defaulting to `auto`:
- `auto` → `YUYV` if resolution ≤ 640×480, else `MJPG` (the standard rule, used for most runs).
- `yuyv` / `mjpg` → explicit override, used for a small **format-comparison sub-experiment** at 640×480 where both formats are valid at 30 FPS. This isolates the cost of JPEG decode per frame (~6 extra runs total).
- `H264` is intentionally not exposed — consuming H.264 from a USB webcam would require a hardware-decode GStreamer pipeline that's out of scope.

The format is pinned via `CAP_PROP_FOURCC`, validated against the camera's reported capabilities at startup (fail-loudly on mismatch), and both requested and actual format are recorded in each run's `config.json`.

---

## 5. Output streaming — how the annotated video gets to a viewer

A deployed detection pipeline almost always sends the annotated frames somewhere (operator monitor, NVR, dashboard). Three common approaches exist:

| Method | Protocol | Encoder | Viewer | Bandwidth | CPU cost |
|---|---|---|---|---|---|
| **MJPEG over HTTP** | HTTP | per-frame JPEG (CPU) | any web browser | high | medium |
| **H.264 over RTP (software)** | RTP/UDP | libx264 (CPU) | VLC, ffplay | low | **high** |
| **H.264 over RTP (NVENC)** | RTP/UDP | NVENC hardware block | VLC, ffplay | low | very low |

H.264 compresses ~10× better than MJPEG by encoding only the differences between frames (inter-frame prediction) — cheap to decode on any device, but expensive to encode in software. NVENC is a dedicated hardware encoder on the Jetson that produces the same H.264 stream at a fraction of the CPU cost.

**Decision for the benchmark:** all four modes are measured as a benchmark axis (`--output-stream {none, mjpeg_cpu, rtp_h264_sw, rtp_h264_nvenc}`). The publishable comparisons:
- `none → mjpeg_cpu / rtp_h264_*` quantifies the energy cost of streaming output at all.
- `rtp_h264_sw → rtp_h264_nvenc` quantifies the gain from hardware encoding — a Jetson-specific result that an edge-inference paper should report.

The NVENC path is implemented via PyGObject + GStreamer (`appsrc → nvvidconv → nvv4l2h264enc → udpsink`). RTSP *input* (consuming a network camera stream) is out of scope for this milestone — input stays on the local USB webcam.

---

## 6. Capture format and streaming format are independent

A common point of confusion: **the format the webcam sends to the Jetson and the format the Jetson sends to a viewer are two separate stages and can be combined freely.**

```
Webcam ──[capture format]──> Jetson decodes to BGR ──> detect + annotate ──> Jetson encodes ──[streaming format]──> Viewer
```

The Jetson always works with decoded BGR pixels in the middle (it has to, in order to draw bounding boxes). So even if capture and streaming use the same format (e.g. MJPG capture + MJPEG streaming), the pipeline still decodes → annotates → re-encodes. There is no shortcut.

In the benchmark, both axes are set independently:
- **Capture format** is set by `--capture-format {yuyv, mjpg, auto}`. Default `auto` picks by resolution; explicit values are used for the format-comparison sub-experiment.
- **Streaming mode** is set by `--output-stream` — its own axis.

In the energy breakdown, capture format affects the **capture stage** (YUYV ≈ free, MJPG = JPEG decode per frame) and streaming mode affects the **encode stage**. Both are reported separately.