# Energy Inference Benchmarking

Python benchmarking pipeline for modeling and predicting neural-network
inference energy/performance. Current focus is CPU benchmarking, feature
extraction, and robust experiment tracking.

## Start here

- Fast setup and first run: this file
- Detailed workflow and "how not to get lost": `docs/EXPERIMENT_RUNBOOK.md`

## What this project does (current stage)

- Runs inference benchmarks on vision models (currently supports `resnet18`, `resnet50`, `mobilenet_v3_large`, `mobilenet_v3_small`, `googlenet`, `shufflenet_v2_x1_0`, `vgg16`, `ssdlite320_mobilenet_v3_large`, `vit_b_16`, `swin_t`, `yolov8n`)
- Tags each row with `model_task` (`classification` or `detection`) to keep comparisons fair
- Measures latency and FPS
- Extracts static features (parameters, FLOPs, metadata)
- Saves one CSV per run by default
- Tracks all runs in `results/run_index.csv`

## Project structure

```text
energy-inference/
  src/
    energy_inference/
      __init__.py
      benchmarking.py
      features.py
      io_utils.py
      models.py
      pipeline.py            # shared run logic used by all scripts
  scripts/
    bench_cpu.py             # benchmark-only CSV
    extract_features.py      # features-only CSV
    run_full.py              # merged benchmark+features CSV
    train_energy_model.py    # simple baseline energy predictor training
    predict_energy.py        # CLI for running predictions with trained energy model
    test_predictor_pipeline.py # interactive terminal loop for predictor testing
    run_experiments_csv.py   # run multiple experiments from CSV
    plot_results.py          # plot one metric vs swept variable
    view_camera.py           # live webcam preview (/dev/videoX)
    live_detect_ssdlite.py   # live webcam object detection with SSDLite
  benchmarks/
    bench.py                 # legacy wrapper
    benchmark.py             # legacy wrapper
    bench.ipynb
  configs/
  data/
    raw/
    processed/
    training_data/           # curated CSVs used for energy model training
  notebooks/
  results/
    run_index.csv            # one row per CLI execution
    runs/                    # default output location (one file per run)
    models/                  # serialized prediction models (e.g., energy predictor)
  requirements.txt
```

## Environment setup

Install dependencies in your conda environment:

```bash
conda activate energy-inference
pip install -r requirements.txt
```

Or run commands without activating:

```bash
conda run -n energy-inference python scripts/run_full.py --help
```

## Jetson (JP6.2) PyTorch setup

On Jetson/aarch64, `pip install -r requirements.txt` intentionally skips
`torch`/`torchvision`. Install Jetson-compatible PyTorch wheels after that:

```bash
conda activate energy-inference

# Optional cleanup if you attempted generic PyPI torch installs before
python -m pip uninstall -y torch torchvision torchaudio

# System dependency
sudo apt-get update
sudo apt-get install -y libopenblas-dev

# Required by newer NVIDIA PyTorch builds
wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.6
bash ./install_cusparselt.sh

# Install JetPack 6.2 compatible wheels
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Minimal quick start

Run one merged sweep (recommended default):

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_baseline_model_sweep \
  --notes "first clean baseline"
```

This will:
- create a run-specific CSV in `results/runs/`
- add one run entry in `results/run_index.csv`
- print the exact `run_id` and output path

Precision sweep example:

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep precision \
  --model resnet18 \
  --precisions fp32 fp16 bf16 \
  --experiment cpu_resnet18_precision_sweep \
  --notes "precision scaling baseline"
```

## Scripts and when to use them

- `scripts/run_full.py`
  - use when you want one merged table per run
  - includes: params/FLOPs + latency/FPS + metadata
  - sweep axes: `model`, `batch`, `resolution`, `precision`
  - precision list arg for precision sweep: `--precisions fp32 fp16 bf16`
- `scripts/bench_cpu.py`
  - use when you only want runtime metrics
  - sweep axes: `model`, `batch`, `resolution`, `precision`
  - precision list arg for precision sweep: `--precisions fp32 fp16 bf16`
- `scripts/extract_features.py`
  - use when you only want model/config features
  - sweep axes: `model`, `batch`, `resolution`, `precision`
  - precision list arg for precision sweep: `--precisions fp32 fp16 bf16`
- `scripts/run_full_cartesian.py`
  - use when you want exhaustive Cartesian benchmarking across model, batch, resolution, and precision
  - defaults:
    - models: `resnet18 resnet50 mobilenet_v3_large mobilenet_v3_small googlenet shufflenet_v2_x1_0 vgg16 vit_b_16 swin_t ssdlite yolo`
    - batches: `1 2 4`
    - resolutions: `160 192 224 256 320 384 448 512 576 640`
    - precisions: `fp32 fp16 bf16`
  - power trace cleanup:
    - by default, deletes `*_power_trace.csv` after integrating energy into the final CSV
    - pass `--keep-power-trace` to preserve traces
  - resume support:
    - pass `--resume-csv <existing_csv>` to skip already present combinations and continue
    - pass `--rerun-failed` with resume mode to retry rows previously marked as `failed`
- `scripts/plot_results.py`
  - use when you want a quick plot from one run CSV
  - auto-detects x-axis from `sweep_param`
  - core usage examples:
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --y latency_ms`
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --plot-latency-fps`
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --plot-energy`
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --summary-energy`
  - run-group plotting in one command:
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --y latency_ms`
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --plot-latency-fps`
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --plot-energy`
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --summary-energy`
  - flags (see `python scripts/plot_results.py --help` for full help):
    - **input / selection**
      - `--input INPUT`: plot from a single run CSV.
      - `--run-dir RUN_DIR`: plot all CSVs in a run-group directory (mutually exclusive with `--input`).
    - **metric / plot type**
      - `--y Y`: metric column for a simple sweep plot (default: `latency_ms`); e.g. `latency_ms`, `fps`, `flops_total_strict`, `macs_total`, `power_total_W`.
      - `--plot-latency-fps`: dual‑axis plot of `latency_ms` and `fps` vs sweep variable.
      - `--plot-energy`: plot `energy_cpu_J`, `energy_gpu_J`, and `energy_io_J` together vs sweep variable.
      - `--summary-energy`: 3‑row stacked figure: total energy, energy per inference, and energy per FLOP.
    - **energy normalization (only with `--plot-energy` / `--summary-energy`)**
      - `--per-sample-energy`: divide `energy_*_J` by `batch` to show energy per inference.
      - `--per-flop-energy`: divide `energy_*_J` by `flops_total_strict` (or `flops_total` on legacy CSVs) to show energy per FLOP.
    - **axes / filtering**
      - `--log-x`: log scale for x‑axis.
      - `--log-y`: log scale for y‑axis (when applicable).
      - `--include-failed`: include rows with `status != "ok"` (by default, only successful rows are plotted when status is present).
    - **output / display**
      - `--output OUTPUT`: explicit output image path when using `--input`.
      - `--output-dir OUTPUT_DIR`: output directory when using `--run-dir` (default: `results/plots/<run_dir_name>`).
      - `--title TITLE`: optional plot title.
      - `--show`: open an interactive window in addition to saving the PNG.
      - `--dpi DPI`: output figure DPI (default: `150`).
- `scripts/plot_training_data_insights.py`
  - use when you want broader training-data analytics (plots + summary CSV tables) from `data/training_data/`
  - cartesian-compatible by default:
    - treats `sweep_param=cartesian` as valid for batch/resolution/precision analyses
    - also remains compatible with legacy single-axis sweep CSVs
  - example:
    - `python scripts/plot_training_data_insights.py --data-dir data/training_data --out-dir results/plots`
- `scripts/view_camera.py`
  - use when you want a quick live preview from a webcam (`/dev/videoX`)
  - default device: `0` (maps to `/dev/video0`)
  - quit key: press `q` in the preview window
  - example:
    - `python scripts/view_camera.py --device 0 --width 1280 --height 720 --fps 30`
- `scripts/live_detect_ssdlite.py`
  - use when you want live object detection from webcam frames with selectable models
  - supported detection models (`--model`):
    - `ssdlite320_mobilenet_v3_large`
    - `fasterrcnn_mobilenet_v3_large_320_fpn`
    - `fasterrcnn_resnet50_fpn_v2`
    - `retinanet_resnet50_fpn_v2`
    - `fcos_resnet50_fpn`
    - `yolov8n`
  - tunable hyperparameters:
    - `--score-threshold`
    - `--max-detections`
    - `--yolo-imgsz` and `--iou-threshold` (YOLO-specific)
  - works in SSH/headless mode by printing detections to terminal
  - add `--show` for an annotated preview window when running on a GUI desktop session
  - supports two remote streaming modes:
    - MJPEG over HTTP (`--serve-mjpeg`) for browser viewing over SSH port-forward
    - RTP/H264 (`--stream-rtp`) for `ffplay` workflows
  - example (headless):
    - `python scripts/live_detect_ssdlite.py --device 0 --model ssdlite320_mobilenet_v3_large --score-threshold 0.5`
  - example (YOLO headless):
    - `python scripts/live_detect_ssdlite.py --device 0 --model yolov8n --score-threshold 0.35 --iou-threshold 0.45 --yolo-imgsz 640`
  - example (GUI):
    - `python scripts/live_detect_ssdlite.py --device 0 --show`
  - example (Mac over SSH, live video in browser):
    - on Mac (new terminal): `ssh -L 8080:127.0.0.1:8080 <user>@<jetson-ip>`
    - on Jetson (inside SSH): `python scripts/live_detect_ssdlite.py --device 0 --serve-mjpeg --mjpeg-host 127.0.0.1 --mjpeg-port 8080`
    - open `http://127.0.0.1:8080` on Mac
  - example (RTP/H264 to ffplay):
    - on Jetson: `python scripts/live_detect_ssdlite.py --device 0 --stream-rtp --rtp-host <mac-ip> --rtp-port 11111 --rtp-sdp /tmp/jetson.sdp`
    - copy the generated `/tmp/jetson.sdp` from Jetson to Mac, then run:
      - `ffplay -protocol_whitelist file,udp,rtp -i jetson.sdp`
  - live power + energy telemetry while detecting:
    - `python scripts/live_detect_ssdlite.py --device 0 --stream-rtp --rtp-host <mac-ip> --rtp-port 11111 --rtp-sdp /tmp/jetson.sdp --enable-energy --ina-hz 1000 --ina-hw all`
  - step-by-step setup guide: `docs/CAMERA_STREAMING_JETSON_TO_MAC.md`
- `scripts/train_energy_model.py`
  - use when you want to train a **simple baseline predictor** for energy consumption from existing run CSVs
  - workflow:
    - run your normal experiments (e.g., with `scripts/run_full.py` or `scripts/run_experiments_csv.py`) which produce CSVs under `results/runs/`
    - manually copy/move the CSVs you want to use for training into `data/training_data/`
      - for example:
        - `cp results/runs/<run_group>/row003_full_jetson_orin_resnet18_resolution_sweep.csv data/training_data/20260304_resolution_sweep.csv`
    - run:
      - `python scripts/train_energy_model.py`
  - key options:
    - `--separate-by model_task|model`
      - `model_task` (default): one regressor per `classification|detection`
      - `model`: one regressor per model name (e.g., `resnet18`, `yolo`, `ssdlite`)
    - `--include-model-feature`
      - adds model one-hot columns when training grouped by `model_task`
      - ignored when `--separate-by model` (model identity is already separated)
  - behavior:
    - automatically uses **all CSVs** in `data/training_data/` (concatenated) for training
    - trains **histogram gradient boosting regressors** (via `scikit-learn`) to predict `energy_cpu_J` from static / hyperparameter-derived features only (no measured latency):
      - numeric: `flops_total`, `batch`, `resolution`
      - categorical (one-hot): `precision` (and optionally `model` with `--include-model-feature`)
      - grouping split controlled by `--separate-by`:
        - one regressor per `model_task` or per `model`
    - applies `log1p` target transform during training and `expm1` at inference, then clips predictions at `>= 0` for physical plausibility
    - prints basic evaluation metrics on a held-out test split:
      - R², MAE (J), and the learned coefficients / intercept
    - saves a serialized model payload under `results/models/energy_cpu_linear.joblib` containing:
      - grouping metadata (`group_by`)
      - grouped estimators (`models_by_group`)
      - grouped feature schemas (`feature_names_by_group`)
      - target name (`energy_cpu_J`) and target transform metadata
  - example of using the saved model for inference in Python:
    - ```python
      from joblib import load

      payload = load("results/models/energy_cpu_linear.joblib")
      models_by_task = payload.get("models_by_task")
      feature_names_by_task = payload.get("feature_names_by_task")
      model = models_by_task["classification"] if models_by_task else payload["model"]
      feature_names = (
          feature_names_by_task["classification"]
          if feature_names_by_task
          else payload["feature_names"]
      )  # e.g., ["flops_total", "batch", "resolution", "model_resnet18", "precision_fp16", ...]
      target_transform = payload.get("target_transform", "none")

      # X_new must be a 2D array / DataFrame with these columns, in this order.
      import pandas as pd
      row = {"flops_total": 3.8e10, "batch": 1, "resolution": 640}
      for f in feature_names:
          if f.startswith("model_"):
              row[f] = 1.0 if f == "model_resnet18" else 0.0
          if f.startswith("precision_"):
              row[f] = 1.0 if f == "precision_fp16" else 0.0
      X_new = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0.0)
      import math
      y_pred = float(model.predict(X_new)[0])
      if target_transform == "log1p":
          y_pred = math.expm1(y_pred)
      y_pred = max(y_pred, 0.0)
      print("Predicted energy_cpu_J:", y_pred)
      ```
- `scripts/predict_energy.py`
  - use when you want a **small CLI** to query the trained energy predictor
  - requires that you have already run `scripts/train_energy_model.py` and have:
    - `results/models/energy_cpu_linear.joblib`
  - example:
    - ```bash
      python scripts/predict_energy.py \
        --flops-total 3.8004576256e10 \
        --model resnet18 \
        --batch 1 \
        --resolution 640 \
        --precision fp16 \
        --model-task classification
      ```
  - behavior:
    - loads the Joblib payload from `results/models/energy_cpu_linear.joblib` (or `--model-path`)
    - constructs a single-row input with features:
      - `flops_total`, `batch`, `resolution`, and model/precision one-hot fields expected by the saved model schema
    - auto-selects grouped estimator based on saved `group_by` metadata:
      - if grouped by `model`, selects by `--model`
      - if grouped by `model_task`, selects `classification|detection` (auto or `--model-task`)
    - prints the predicted `energy_cpu_J` to stdout
- `scripts/test_predictor_pipeline.py`
  - use when you want an interactive terminal loop to test many predictor inputs quickly
  - prompts one-by-one for:
    - `model`, `flops_total`, `batch`, `resolution`, `precision`, optional `model_task` override
  - supports grouped predictor payloads (`group_by=model` or `group_by=model_task`)
  - example:
    - `python scripts/test_predictor_pipeline.py --model-path results/models/energy_cpu_linear.joblib`
- `scripts/run_experiments_csv.py`
  - use when you want to run many experiments from a CSV template
  - example:
    - `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv`
  - dry run (validate only):
    - `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv --dry-run`
- `scripts/compare_flops_methods.py`
  - use when you want to compare `fvcore` against `THOP` on the same models/shapes
  - example:
    - `python scripts/compare_flops_methods.py --device cpu --models resnet18 resnet50 --batches 1 8 --resolutions 224 640`
  - output:
    - writes `results/flops/compare_fvcore_vs_thop.csv` by default with:
      - `fvcore_flops`
      - `thop_macs`
      - `thop_flops_x2`
      - percent-difference columns to quickly assess mismatch magnitude
  - each execution creates one run-group directory under `results/runs/` and stores all child CSVs there

All three support run tracking flags:
- `--experiment`
- `--run-id`
- `--notes`
- `--out`
- `--append`

## Result tracking model

- Every data row includes:
  - `run_id`, `experiment`, `notes`, and measurement columns
- Every CLI execution appends one row to `results/run_index.csv` with:
  - `run_id`, `timestamp`, `mode`, `output_path`, `device`, `sweep_param`, `experiment`, `notes`

This is the key mechanism that prevents confusion when you run many experiments.

## FLOPs backend policy

- FLOPs/MACs are computed with `THOP` (`ultralytics-thop`).
- `macs_total` stores the raw THOP MAC count.
- `flops_total_strict` stores strict FLOPs (`2 * macs_total`).
- `flops_total` is kept as a backward-compatible alias to `flops_total_strict`.
- `unsupported_ops_count` is kept and set to `0` for backward-compatible CSV schema.

## Legacy command

```bash
python benchmarks/bench.py --sweep model --models resnet18 resnet50
```

## Detailed docs

For detailed commands, naming conventions, and reproducible workflow checklists,
see `docs/EXPERIMENT_RUNBOOK.md`.

You can also pass a CSV directly to `run_full.py`:

```bash
python scripts/run_full.py --experiments-csv configs/experiments_example.csv
```

Validate CSV without running:

```bash
python scripts/run_full.py --experiments-csv configs/experiments_example.csv --dry-run
```

Jetson AGX Orin profile (GPU):

```bash
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv --dry-run
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv
```

