# AGENTS GUIDE (energy-inference)

This file is the source of truth for AI/code agents working in this repository.

## 1) Repository orientation

- Core package: `src/energy_inference/`
  - `models.py`: model factory (currently ResNet18/ResNet50)
  - `benchmarking.py`: timed inference loop
  - `features.py`: params/FLOPs helpers
  - `pipeline.py`: shared sweep runner for `bench` / `features` / `full`
  - `plotting.py`: plotting utilities
  - `experiment_runner.py`: CSV-driven experiment runner
- CLI scripts: `scripts/`
  - `run_full.py`
  - `bench_cpu.py`
  - `extract_features.py`
  - `run_experiments_csv.py`
  - `plot_results.py`
- Experiment presets: `configs/*.csv`
- Results output: `results/` (ignored in git)
- Documentation: `README.md`, `docs/EXPERIMENT_RUNBOOK.md`

## 2) Environment and install

- Python target: `>=3.10`.
- Typical setup:
  1. `python3 -m venv .venv` (or conda env)
  2. activate env
  3. `pip install -r requirements.txt`

### Jetson note (aarch64)

- `requirements.txt` intentionally skips `torch`/`torchvision` on `aarch64`.
- Install JetPack-compatible `torch`/`torchvision` separately.
- If torch import fails with missing CUDA libs, fix env/runtime libs before code changes.

## 3) Canonical run commands

- Single merged sweep:
  - `python scripts/run_full.py --sweep model --models resnet18 resnet50 --experiment <name> --notes "<text>"`
- CSV-driven runs:
  - dry-run: `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv --dry-run`
  - execute: `python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv`
- Plot single run CSV:
  - `python scripts/plot_results.py --input results/runs/<file>.csv --y latency_ms`
- Plot full run-group dir:
  - `python scripts/plot_results.py --run-dir results/runs/<group_dir> --y latency_ms`

## 4) Current design rules

- Keep code research-oriented and simple; avoid overengineering.
- Prefer modular helpers in `src/energy_inference/` over duplicating logic in scripts.
- Keep CLIs `argparse`-driven.
- Preserve run tracking fields:
  - `run_id`, `experiment`, `notes`
- CSV experiment semantics:
  - if `sweep=model`, `models` is the true sweep list (`model` can be empty)
  - if `sweep=batch` or `sweep=resolution`, `model` is fixed base model

## 5) GPU timing and behavior

- For GPU timing correctness, synchronize around timed regions (`torch.cuda.synchronize()`).
- Warmup should remain outside measured timing windows.
- Preserve `status`/`error_msg` behavior so sweeps continue on partial failures.

## 6) FLOPs policy (explicit project choice)

- Unsupported-op warnings are silenced.
- `unsupported_ops_count` is intentionally forced to `0` in CSV outputs.

## 7) Data/output policy

- Never commit generated artifacts (`results/**`, plots, traces).
- Keep config templates (`configs/*.csv`) tracked.
- New output paths should remain under `results/` unless explicitly needed elsewhere.

## 8) Editing and quality checklist

Before finishing changes:
1. Run the relevant CLI help/smoke command.
2. Confirm no lints/regressions in modified files.
3. Update `README.md` and/or `docs/EXPERIMENT_RUNBOOK.md` when workflow changes.
4. Keep behavior backward compatible unless explicitly requested otherwise.

