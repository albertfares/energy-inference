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
  notebooks/
  results/
    run_index.csv            # one row per CLI execution
    runs/                    # default output location (one file per run)
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
  - example:
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --y latency_ms`
    - `python scripts/plot_results.py --input results/runs/<your_run>.csv --plot-latency-fps`
  - run-group plotting in one command:
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --y latency_ms`
    - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --plot-latency-fps`
- `scripts/run_experiments_csv.py`
  - use when you want to run many experiments from a CSV template
  - example:
    - `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv`
  - dry run (validate only):
    - `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv --dry-run`
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

## FLOPs warning policy

- Unsupported-op warnings from `fvcore` are explicitly silenced.
- `unsupported_ops_count` is explicitly set to `0` in saved CSVs by design.
- This keeps terminal output clean during sweeps.

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

