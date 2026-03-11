# Energy Inference Benchmarking

Python benchmarking pipeline for modeling and predicting neural-network
inference energy/performance. Current focus is CPU benchmarking, feature
extraction, and robust experiment tracking.

## Start here

- Fast setup and first run: this file
- Detailed workflow and "how not to get lost": `docs/EXPERIMENT_RUNBOOK.md`

## What this project does (current stage)

- Runs inference benchmarks on vision models (currently `resnet18`, `resnet50`)
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
    run_full_cpu.py          # merged benchmark+features CSV
    train_energy_model.py    # simple baseline energy predictor training
    predict_energy.py        # CLI for running predictions with trained energy model
    run_experiments_csv.py   # run multiple experiments from CSV
    plot_results.py          # plot one metric vs swept variable
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
conda run -n energy-inference python scripts/run_full_cpu.py --help
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
conda run -n energy-inference python scripts/run_full_cpu.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_baseline_model_sweep \
  --notes "first clean baseline"
```

This will:
- create a run-specific CSV in `results/runs/`
- add one run entry in `results/run_index.csv`
- print the exact `run_id` and output path

## Scripts and when to use them

- `scripts/run_full_cpu.py`
  - use when you want one merged table per run
  - includes: params/FLOPs + latency/FPS + metadata
- `scripts/bench_cpu.py`
  - use when you only want runtime metrics
- `scripts/extract_features.py`
  - use when you only want model/config features
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
- `scripts/train_energy_model.py`
  - use when you want to train a **simple baseline predictor** for energy consumption from existing run CSVs
  - workflow:
    - run your normal experiments (e.g., with `scripts/run_full_cpu.py` or `scripts/run_experiments_csv.py`) which produce CSVs under `results/runs/`
    - manually copy/move the CSVs you want to use for training into `data/training_data/`
      - for example:
        - `cp results/runs/<run_group>/row003_full_jetson_orin_resnet18_resolution_sweep.csv data/training_data/20260304_resolution_sweep.csv`
    - run:
      - `python scripts/train_energy_model.py`
  - behavior:
    - automatically uses **all CSVs** in `data/training_data/` (concatenated) for training
    - currently trains a **linear regression model** (via `scikit-learn`) to predict `energy_cpu_J` from **static / hyperparameter-derived numeric features only** (no measured latency):
      - `flops_total`, `batch`, `resolution`
    - prints basic evaluation metrics on a held-out test split:
      - R², MAE (J), and the learned coefficients / intercept
    - saves a serialized model payload under `results/models/energy_cpu_linear.joblib` containing:
      - the trained model
      - the feature name list
      - the target name (`energy_cpu_J`)
  - example of using the saved model for inference in Python:
    - ```python
      from joblib import load

      payload = load("results/models/energy_cpu_linear.joblib")
      model = payload["model"]
      feature_names = payload["feature_names"]  # ["flops_total", "batch", "resolution"]

      # X_new must be a 2D array / DataFrame with these columns, in this order.
      import pandas as pd
      X_new = pd.DataFrame(
          [
              {"flops_total": 3.8e10, "batch": 1, "resolution": 640},
          ]
      )[feature_names]
      y_pred = model.predict(X_new)
      print("Predicted energy_cpu_J:", y_pred[0])
      ```
- `scripts/predict_energy.py`
  - use when you want a **small CLI** to query the trained energy predictor
  - requires that you have already run `scripts/train_energy_model.py` and have:
    - `results/models/energy_cpu_linear.joblib`
  - example:
    - ```bash
      python scripts/predict_energy.py \
        --flops-total 3.8004576256e10 \
        --batch 1 \
        --resolution 640
      ```
  - behavior:
    - loads the Joblib payload from `results/models/energy_cpu_linear.joblib` (or `--model-path`)
    - constructs a single-row input with features:
      - `flops_total`, `batch`, `resolution`
    - prints the predicted `energy_cpu_J` to stdout
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

You can also pass a CSV directly to `run_full_cpu.py`:

```bash
python scripts/run_full_cpu.py --experiments-csv configs/experiments_example.csv
```

Validate CSV without running:

```bash
python scripts/run_full_cpu.py --experiments-csv configs/experiments_example.csv --dry-run
```

Jetson AGX Orin profile (GPU):

```bash
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv --dry-run
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv
```

