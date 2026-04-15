# Camera Pipeline Benchmark

Research-grade benchmarking tool that measures energy, latency, and FPS of a **real camera detection pipeline** on a Jetson device — from `cap.read()` to the final encoded output frame.

Unlike the synthetic sweep (which times only `model(tensor)`), this tool measures the full deployment cost: capture, preprocessing, inference, postprocessing, and output streaming. It also provides a per-stage energy breakdown showing which pipeline component consumes which share of the total power draw.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Package layout](#2-package-layout)
3. [Running a single benchmark](#3-running-a-single-benchmark)
4. [Running a grid sweep](#4-running-a-grid-sweep)
5. [Generating plots](#5-generating-plots)
6. [Output file reference](#6-output-file-reference)
7. [Design notes](#7-design-notes)
8. [Pre-flight checklist (Jetson)](#8-pre-flight-checklist-jetson)

---

## 1. Prerequisites

All commands are run from the **project root** with `PYTHONPATH=src` set.

### Python dependencies

Everything already in `requirements.txt` plus:

```bash
# Already required by the project
pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib

# For NVENC streaming only (Jetson):
conda install -c conda-forge pygobject gst-python
```

### Hardware

- Logitech webcam (or any V4L2 camera) at `/dev/video0`
- INA3221 power sampler binary at `src/energy_inference/tools/sample_ina3221` (compiled from the C++ source in that directory)
- For `rtp_h264_nvenc` streaming: JetPack with GStreamer NVIDIA plugins (`nvv4l2h264enc`)

### Environment variable for NVENC

If using `--output-stream rtp_h264_nvenc`, the conda-forge GStreamer needs to find the system NVIDIA plugins. The tool sets this automatically if unset:

```bash
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
```

---

## 2. Package layout

```
src/camera_bench/
├── __init__.py                  Package init
│
├── metrics.py                   Pure math — no I/O, no torch
├── power.py                     INA3221 power monitor wrapper
├── capture.py                   Webcam open/configure helpers
├── models.py                    Model loading registry
├── detection.py                 Per-stage detection with cuda.synchronize()
├── results.py                   Results directory layout and writers
│
├── pipeline.py                  Main per-frame benchmark loop
├── cli.py                       argparse entry point (single runs)
├── sweep.py                     Grid sweep runner (subprocess per run)
│
└── output_streaming/
    ├── __init__.py              Factory function get_streamer()
    ├── base.py                  OutputStreamer ABC
    ├── none_stream.py           No-op: annotation + encoding skipped
    ├── mjpeg_cpu.py             HTTP MJPEG via cv2.imencode
    ├── rtp_h264_sw.py           RTP via ffmpeg + libx264 (CPU)
    └── rtp_h264_nvenc.py        RTP via GStreamer + nvv4l2h264enc (NVENC)

scripts/
└── plot_camera_bench.py         Five benchmark plots (matplotlib only)

tests/
└── test_camera_bench_metrics.py 22 unit tests for metrics.py
```

### File descriptions

| File | What it does |
|---|---|
| `metrics.py` | `trapz_energy_j` — trapezoidal integration of a power trace over a time window. `attribute_stage_energy` — aligns INA3221 samples with per-frame stage timestamps to compute energy per stage. `percentile_stats`, `compute_fps_stats`, `compute_stage_latency_stats` — pure aggregation with no side effects. |
| `power.py` | `PowerMonitor` wraps `INA3221Sampler`. Provides `start()`, `stop()`, `load_power_trace()`, and `check_clock_alignment_ns()` — which verifies that Python's `time.monotonic_ns()` and the sampler's `CLOCK_MONOTONIC` are within 1 ms of each other. |
| `capture.py` | `open_camera()` opens a V4L2 device and reads back the actual negotiated resolution. `validate_camera_resolution()` prints a warning if the camera silently fell back to a different mode. |
| `models.py` | `load_detector()` returns a unified descriptor dict for any supported model. Handles torchvision models (SSDLite, FasterRCNN variants, RetinaNet, FCOS) and ultralytics YOLO. `_apply_precision()` casts the model to fp16/bf16 if requested. |
| `detection.py` | `run_staged_detection()` runs one frame and returns a `{stage: (t_start_ns, t_end_ns)}` dict alongside boxes/labels/scores. Torchvision path: `preprocess → infer → postprocess → filter`, each delimited with `torch.cuda.synchronize()`. YOLO path: `infer_fused` covers the entire `model.predict()` call (pre+infer+post are fused internally); the individual stage entries are set to `(0, 0)`. |
| `results.py` | `ResultsDir` manages one run directory. Buffers frame rows in memory during the run, then flushes to `frames.csv` atomically. Writes `config.json`, `stage_energy.csv`, `power_trace.csv`, and `summary.json`. `collect_env_info()` records torch/torchvision/ultralytics/JetPack versions. |
| `pipeline.py` | Orchestrates the full run: load model → open camera → open streamer → warmup (30 frames, not timed) → start INA3221 → timed loop → stop sampler → compute energy attribution → write outputs. Calls `_print_summary()` at the end. |
| `cli.py` | Single-run entry point. Parses all CLI flags, builds a `cfg` dict, calls `pipeline.run_benchmark()`. Supports `--repeat N` with `--cooldown-s`. Requires `--bench-mode` to be set. |
| `sweep.py` | Grid sweep runner. Enumerates a configuration grid (built-in or JSON file), runs each config as a clean subprocess (to prevent PyTorch/CUDA state leakage), waits `--cooldown-s` between runs, and appends every result to `sweep_summary.csv`. Supports `--dry-run` to print the grid without executing and `--resume` to skip already-completed configs. |
| `output_streaming/base.py` | `OutputStreamer` ABC. Three methods: `open()`, `push(frame_bgr)`, `close()`. The boolean property `needs_annotation` tells the main loop whether to call `build_annotated_frame()` at all — the `none` mode returns `False`, saving annotation-drawing cost. |
| `output_streaming/none_stream.py` | No-op. `push()` does nothing; `needs_annotation` is `False`. The lowest-overhead baseline. |
| `output_streaming/mjpeg_cpu.py` | `ThreadingHTTPServer` + `cv2.imencode(..., JPEG)` served at `http://host:port/stream.mjpg`. One JPEG per frame. |
| `output_streaming/rtp_h264_sw.py` | Spawns an `ffmpeg` subprocess, pipes raw BGR frames into stdin. Uses `libx264 -preset ultrafast -tune zerolatency`. Writes an SDP file for the receiver. |
| `output_streaming/rtp_h264_nvenc.py` | Builds a GStreamer pipeline (`appsrc → videoconvert → nvvidconv → nvv4l2h264enc → rtph264pay → udpsink`) using PyGObject. Pushes BGR frames as `Gst.Buffer` objects with proper PTS. `smoke_test_nvenc()` runs a one-shot test pipeline before touching the camera; the benchmark refuses to start if it fails. |
| `scripts/plot_camera_bench.py` | Five plots from a sweep directory: (1) FPS achieved vs target, (2) energy/frame by model×precision, (3) stage energy stacked bar, (4) streaming overhead comparison, (5) sustained-run power+FPS timeline. |
| `tests/test_camera_bench_metrics.py` | 22 unit tests for `metrics.py`. Covers constant/ramp/sub-interval integration, NaN handling, out-of-window extrapolation, multi-stage/multi-rail attribution, FPS and latency stats. Run without any hardware. |

---

## 3. Running a single benchmark

```bash
# Basic 60-second run, yolov8n, no output streaming
PYTHONPATH=src python -m camera_bench.cli \
  --bench-mode \
  --model yolov8n \
  --duration-s 60 \
  --stage-energy \
  --output-stream none

# fp16, 640×480, 30 FPS target, RTP NVENC streaming to a receiver at 192.168.1.10
PYTHONPATH=src python -m camera_bench.cli \
  --bench-mode \
  --model yolov8n \
  --precision fp16 \
  --width 640 --height 480 \
  --target-fps 30 \
  --output-stream rtp_h264_nvenc \
  --stream-host 192.168.1.10 \
  --stream-port 11111 \
  --duration-s 120 \
  --out-dir results/camera_bench

# Three repeats with 30-second cooldown between them
PYTHONPATH=src python -m camera_bench.cli \
  --bench-mode \
  --model ssdlite320_mobilenet_v3_large \
  --precision fp16 \
  --repeat 3 \
  --cooldown-s 30 \
  --duration-s 120
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--bench-mode` | off | Required to activate benchmark behaviour |
| `--model` | `ssdlite320_mobilenet_v3_large` | Detection model |
| `--precision` | `fp32` | `fp32`, `fp16`, or `bf16` |
| `--cpu` | off | Force CPU even if CUDA is available |
| `--width` / `--height` | 640 / 480 | Requested camera resolution |
| `--fps` | 30 | Requested camera FPS |
| `--duration-s` | 120 | Length of the timed window |
| `--warmup-frames` | 30 | Frames to discard before timing starts |
| `--target-fps` | 0 | Pace the loop to this FPS (0 = unbounded) |
| `--output-stream` | `none` | `none`, `mjpeg_cpu`, `rtp_h264_sw`, `rtp_h264_nvenc` |
| `--stream-host` | `127.0.0.1` | Destination for RTP modes |
| `--stream-port` | 11111 | Destination port |
| `--stream-bitrate` | 2000000 | H.264 target bitrate (bps) |
| `--stage-energy` / `--no-stage-energy` | on | Per-stage INA3221 attribution |
| `--enable-energy` / `--no-energy` | on | INA3221 power sampling |
| `--ina-hz` | 1000 | Sampler frequency (Hz) |
| `--sampler-exe` | `src/energy_inference/tools/sample_ina3221` | Path to INA3221 binary |
| `--repeat` | 1 | Number of back-to-back repeats |
| `--cooldown-s` | 30 | Idle time between repeats |
| `--out-dir` | `results/camera_bench` | Base results directory |
| `--run-name` | `auto` | Run subdirectory name (auto-generated if `auto`) |

---

## 4. Running a grid sweep

```bash
# Dry run: print all 144 + 12 configs without executing
PYTHONPATH=src python -m camera_bench.sweep \
  --grid default \
  --dry-run

# Real overnight sweep
PYTHONPATH=src python -m camera_bench.sweep \
  --grid default \
  --out-dir results/camera_bench/sweep_$(date +%Y%m%d) \
  --repeats 3 \
  --duration-s 120 \
  --cooldown-s 30

# Streaming-comparison sub-grid only (12 runs)
PYTHONPATH=src python -m camera_bench.sweep \
  --grid streaming \
  --out-dir results/camera_bench/streaming_comparison \
  --stream-host 192.168.1.10

# Minimal 4-run sanity check
PYTHONPATH=src python -m camera_bench.sweep --grid minimal

# Resume an interrupted sweep (skips runs whose run_dir already has summary.json)
PYTHONPATH=src python -m camera_bench.sweep \
  --grid default \
  --out-dir results/camera_bench/sweep_20260415 \
  --resume

# Custom grid from a JSON file
PYTHONPATH=src python -m camera_bench.sweep --grid my_grid.json
```

**Custom grid JSON format:**

```json
{
  "model":         ["yolov8n", "ssdlite320_mobilenet_v3_large"],
  "width":         [640],
  "height":        [480],
  "precision":     ["fp32", "fp16"],
  "target_fps":    [0, 30],
  "output_stream": ["none", "rtp_h264_nvenc"]
}
```

Every combination of the listed values is run × `--repeats`.

**Built-in grids:**

| Grid name | Configs | Est. time (3 repeats, 2 min/run) |
|---|---|---|
| `default` | 3 models × 2 resolutions × 2 precisions × 2 FPS × 2 streams = 48 | ~6 h |
| `streaming` | 1 model × 4 streaming modes = 4 | ~25 min |
| `minimal` | 1 config | ~7 min |

**Sweep output structure:**

```
results/camera_bench/sweep_20260415/
  sweep_summary.csv           One row per completed run (headline numbers)
  sweep_summary.json          Same data as JSON
  sweep_failures.csv          Rows for crashed/timed-out runs
  <run_name>/                 One directory per run (see §6)
    ...
```

---

## 5. Generating plots

```bash
python scripts/plot_camera_bench.py \
  --sweep-dir results/camera_bench/sweep_20260415

# Specific plots only
python scripts/plot_camera_bench.py \
  --sweep-dir results/camera_bench/sweep_20260415 \
  --plots stages streaming

# Custom output directory
python scripts/plot_camera_bench.py \
  --sweep-dir results/camera_bench/sweep_20260415 \
  --out-dir results/plots/camera_bench
```

Plots are saved as PNG in `<sweep-dir>/plots/` by default.

| File | Description |
|---|---|
| `fps_vs_target.png` | Line chart: achieved mean FPS vs target FPS, one line per model. Shows whether each model can keep up at 15/30 FPS. |
| `energy_per_frame.png` | Bar chart: energy per frame (mJ) grouped by model, colored by precision. Filtered to `output_stream=none` to isolate inference cost. |
| `stage_energy_breakdown.png` | Stacked bar: share of total energy per pipeline stage, per model. The headline figure for the "where does the energy go" question. |
| `streaming_overhead.png` | Two panels for the reference config (yolov8n fp16): left = energy/frame by streaming mode; right = stage breakdown per mode. Shows the cost of output delivery and the NVENC saving vs libx264. |
| `sustained_timeline.png` | Power (W) and FPS over time for the longest run in the sweep (or `sustained/` subdirectory). Used to detect thermal throttling. |

---

## 6. Output file reference

Each single run produces:

```
results/camera_bench/<run_name>/
  config.json          Full configuration: all CLI args + actual camera resolution
                       + env (torch/torchvision/ultralytics/JetPack versions, hostname)
  frames.csv           One row per timed frame — see columns below
  power_trace.csv      Raw INA3221 samples for the timed window (mono_ns, *_power_mW)
  stage_energy.csv     Per (stage, rail): energy_j, mean_power_w, share_pct
  summary.json         Headline numbers — see structure below
  log.txt              stdout/stderr from the run
```

### `frames.csv` columns

| Column | Description |
|---|---|
| `frame_idx` | 0-based frame counter |
| `t_capture_start_ns` / `t_capture_end_ns` | `cap.read()` wall-clock window (CLOCK_MONOTONIC ns) |
| `t_preprocess_*` | BGR→tensor conversion + `.to(device)` |
| `t_infer_*` | `model([image])` (torchvision) |
| `t_infer_fused_*` | `model.predict()` (YOLO — covers pre+infer+post) |
| `t_postprocess_*` | D2H transfer + output dict extraction (torchvision) |
| `t_filter_*` | Score threshold + max-detection clipping |
| `t_annotate_*` | `build_annotated_frame()` (0 if `output_stream=none`) |
| `t_encode_*` | `streamer.push()` (0 if `output_stream=none`) |
| `n_detections` | Detections kept after filtering |
| `latency_total_ms` | capture start → last encode end |
| `fps_inst` | 1 / (this frame's capture start − previous frame's) |

All timestamps are `CLOCK_MONOTONIC` nanoseconds — the same reference as the INA3221 power trace `mono_ns` column. This is what makes time-aligned energy attribution possible.

### `summary.json` structure

```json
{
  "config": { "model": "yolov8n", "precision": "fp16", ... },
  "duration_s": 120.3,
  "n_warmup": 30,
  "n_timed": 3601,
  "fps": { "mean": 29.5, "p50": 29.8, "p95": 27.2, "min": 18.4, "max": 30.3 },
  "latency_ms": {
    "total": { "mean": 33.8, "p50": 33.5, "p95": 37.1 },
    "per_stage": {
      "capture":    { "mean": 3.2, "p50": 3.1, "p95": 4.0 },
      "infer_fused": { "mean": 21.4, "p50": 21.1, "p95": 24.3 },
      ...
    }
  },
  "energy": {
    "total_j": 248.4,
    "mean_power_w": 2.07,
    "per_rail_j": { "cpu": 70.1, "gpu": 132.0, "io": 46.3 },
    "per_stage_j": { "infer_fused": 149.0, "capture": 18.2, ... },
    "per_stage_pct": { "infer_fused": 60.0, "capture": 7.3, ... },
    "energy_per_frame_j": 0.0690,
    "energy_per_inference_j": 0.0690
  }
}
```

---

## 7. Design notes

### Why subprocess-per-run in the sweep

Each sweep config is run as a fresh `python -m camera_bench.cli` subprocess. This guarantees no PyTorch/CUDA state (allocated tensors, cached kernels, JIT artefacts) leaks between configurations. A shared-process sweep would be faster but would risk cross-contaminating measurements.

### Stage timing and CUDA synchronization

GPU kernel launches are asynchronous. Without `torch.cuda.synchronize()` at the end of each GPU stage, the timestamps would be wrong — the kernel would still be running when the end-time is read, making the inference stage appear shorter and the following stage appear longer.

The torchvision path synchronizes after `preprocess` (tensor uploaded to device), `infer` (forward pass), and `postprocess` (D2H copy). The YOLO path synchronizes after the whole `model.predict()` call.

### YOLO stage fusion

The `ultralytics` `model.predict()` API fuses preprocessing (letterbox resize, normalization), inference, and NMS postprocessing into a single call. Breaking these apart requires calling internal methods that are not stable across versions.

For YOLO, the benchmark therefore reports:
- `infer_fused`: total time for `model.predict()` including pre/infer/post
- `preprocess`, `infer`, `postprocess`: all set to `(0, 0)` — do not interpret these

This is noted in `summary.json` via a console message when the model is loaded.

### Energy attribution methodology

1. The INA3221 sampler runs at 1 kHz and writes `(mono_ns, cpu_power_mW, gpu_power_mW, io_power_mW)` rows.
2. For each timed frame, every stage has a `(t_start_ns, t_end_ns)` interval.
3. For each interval and each rail, `trapz_energy_j` integrates the power trace over `[t_start_s, t_end_s]` using boundary interpolation + trapezoidal rule.
4. Energies are accumulated across all frames to give total stage energy.
5. `idle/other = total_INA3221_energy − sum(attributed_stage_energies)`. This is non-zero because the pipeline sleeps between frames (waiting for the next camera frame at 30 FPS ≈ 33 ms/frame).

The attribution is only as accurate as the clock alignment between Python's `time.monotonic_ns()` and the sampler's `CLOCK_MONOTONIC`. The tool checks this at startup; an offset > 1 ms prints a warning.

### Batch size

This benchmark is batch=1 only. Real-time camera pipelines do not batch frames. Batch > 1 is out of scope here; it is covered by the synthetic sweep in `src/energy_inference/`.

### Output streaming as a measured axis

Almost every deployed camera pipeline transmits the annotated frames somewhere (operator monitor, NVR, dashboard). The `--output-stream` axis measures this cost. The four modes span the space from "no output" to "hardware-encoded RTP", producing the publishable result: "output delivery adds X mJ/frame; NVENC saves Y% vs libx264".

---

## 8. Pre-flight checklist (Jetson)

Before starting an overnight sweep, verify:

1. **Power mode is fixed.**
   ```bash
   sudo nvpmodel -q
   sudo jetson_clocks
   ```
   Record and pin the mode. DVFS switching mid-run corrupts energy data.

2. **INA3221 binary is built.**
   ```bash
   ls -l src/energy_inference/tools/sample_ina3221
   # If missing:
   cd src/energy_inference/tools && make
   ```

3. **Camera supports the requested resolution.**
   ```bash
   v4l2-ctl --list-formats-ext -d /dev/video0 | grep -A2 "640x480"
   ```
   The tool will warn (not abort) if the camera falls back to a different resolution.

4. **No other GPU consumers running.**
   ```bash
   tegrastats  # or watch nvidia-smi (if available)
   ```

5. **NVENC available (if using `rtp_h264_nvenc`).**
   ```bash
   PYTHONPATH=src python3 -c "
   from camera_bench.output_streaming.rtp_h264_nvenc import smoke_test_nvenc
   print('NVENC OK' if smoke_test_nvenc() else 'NVENC FAILED')
   "
   ```

6. **Disk space for power traces.**
   At 1 kHz × ~200 min/night, traces are ~50 MB. For a full 144-run sweep they are negligible. Still worth checking:
   ```bash
   df -h results/
   ```

7. **Run the unit tests** (on any machine, no hardware needed).
   ```bash
   PYTHONPATH=src python3 -m unittest tests/test_camera_bench_metrics.py -v
   ```
