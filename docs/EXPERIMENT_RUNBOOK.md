# Experiment Runbook

This document is the detailed "how to run and stay organized" guide for this
repository. When in doubt, follow this file exactly.

## 1) Core idea

You run one of the scripts in `scripts/`.

Each run gets:
- a `run_id`
- an experiment label (`--experiment`)
- optional notes (`--notes`)
- a CSV output file (usually under `results/runs/`)
- a run registry entry in `results/run_index.csv`

This design makes it easy to recover context later even after many runs.

## 2) Scripts overview

### `scripts/run_full.py` (recommended default)

Produces one merged CSV with:
- model/config metadata
- parameter count
- FLOPs
- latency and FPS
- status/error columns

Use this for most baseline sweeps.

Supported model names include:
- `resnet18`, `resnet50`
- `mobilenet_v3_large`, `mobilenet_v3_small`
- `ssdlite320_mobilenet_v3_large` (aliases: `ssdlite`, `ssdlite320`)
- `vit_b_16` (alias: `vit`)
- `swin_t` (aliases: `swin`, `swin_tiny`)
- `yolov8n` (aliases: `yolo`, `yolov8n.yaml`; requires `ultralytics`)

### `scripts/bench_cpu.py`

Benchmark-only output:
- latency/FPS metrics
- no FLOPs/param extraction

### `scripts/extract_features.py`

Features-only output:
- params/FLOPs/metadata
- no benchmark timing loop

## 3) CLI flags you should remember

### Tracking flags (important)

- `--experiment`  
  Human-readable run group name. Example: `cpu_batch_sweep_resnet18`.

- `--notes`  
  Free text description. Example: `"warmup 50, laptop on battery"`.

- `--run-id`  
  Optional manual ID. Usually leave unset so script auto-generates it.

- `--out`  
  Optional explicit output path. If omitted, default is generated under
  `results/runs/`.

- `--append`  
  Append mode. Use with care. Without `--append`, the system prefers safe
  run-specific files.

### Experiment config flags

- `--device {cpu,cuda}`
- `--sweep {model,batch,resolution,precision}`
- `--models ...`
- `--batches ...`
- `--resolutions ...`
- `--model`, `--batch`, `--resolution` (base values for non-swept dimensions)

Only for benchmark/full:
- `--iters`
- `--warmup`

Only for features/full:
- `--precision`
- `--backend`

## 4) Recommended workflow (copy this)

### Step A: decide a clear experiment name

Use a stable pattern:

`<device>_<target>_<sweep>_<date_or_context>`

Examples:
- `cpu_baseline_model_sweep`
- `cpu_resnet18_batch_sweep_plugged`
- `cpu_resolution_sweep_iters500`

### Step B: run merged pipeline

```bash
conda run -n energy-inference python scripts/run_full.py \
  --device cpu \
  --sweep model \
  --models resnet18 resnet50 \
  --iters 200 \
  --warmup 30 \
  --experiment cpu_baseline_model_sweep \
  --notes "plugged in, no background apps"
```

### Step C: confirm where output went

The script prints:
- `run_id=...`
- `results saved to: ...`

Also check `results/run_index.csv` for traceability.

## 5) Output files and schemas

## `results/run_index.csv`

One row per script execution:
- `run_id`
- `timestamp`
- `mode` (`bench` / `features` / `full`)
- `output_path`
- `device`
- `sweep_param`
- `experiment`
- `notes`

Use this file as your master index.

## Per-run CSVs in `results/runs/`

For `run_full.py`, columns include:
- tracking: `run_id`, `experiment`, `notes`, `timestamp`
- config: `device`, `sweep_param`, `model_family`, `model`, `model_task`, `batch`, `resolution`, `precision`, `backend`, `iters`, `warmup`
- features: `num_params`, `macs_total`, `flops_total`, `flops_total_strict`, `flops_per_sample`, `unsupported_ops_count`
- runtime: `latency_ms`, `fps`, `power_total_W`
- health: `status`, `error_msg`

Use `model_task` to separate analyses:
- `classification`: resnet/mobilenet/vit/swin
- `detection`: ssdlite/yolo-style models

## 6) Multiple sweeps strategy

To avoid confusion:

- Keep one experiment per run command.
- Use explicit notes when changing runtime environment.
- Do not mix unrelated studies under one `--experiment` tag.

Good practice:
- one experiment per hypothesis
- one or more runs per experiment
- compare by filtering `experiment` and/or `run_id`

## 7) Suggested naming convention

For `--experiment`, use:

`<device>_<model_scope>_<sweep>_<extra>`

Examples:
- `cpu_resnets_model_baseline`
- `cpu_resnet18_batch_scaling`
- `cpu_resnet50_resolution_scaling`

For `--notes`, include short context:
- power state (`plugged`, `battery`)
- machine load (`idle`, `other apps running`)
- any unusual settings

## 8) Reproducibility checklist

Before running:
- confirm conda env: `energy-inference`
- close heavy background workloads
- decide fixed `iters/warmup` values
- define experiment tag and notes

After running:
- copy `run_id` into your lab notes/report log
- verify row exists in `results/run_index.csv`
- spot-check output CSV for expected sweep values

## 9) Example command recipes

### Model sweep (default baseline)

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_resnets_model_baseline \
  --notes "standard settings"
```

### Batch sweep on one model

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep batch \
  --model resnet18 \
  --batches 1 2 4 8 16 \
  --experiment cpu_resnet18_batch_scaling \
  --notes "checking throughput trend"
```

### Resolution sweep

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep resolution \
  --model resnet50 \
  --resolutions 160 224 320 384 \
  --experiment cpu_resnet50_resolution_scaling \
  --notes "same iters/warmup as baseline"
```

### Precision sweep

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep precision \
  --model resnet18 \
  --precisions fp32 fp16 bf16 \
  --experiment cpu_resnet18_precision_scaling \
  --notes "same iters/warmup as baseline"
```

### Benchmark-only run

```bash
conda run -n energy-inference python scripts/bench_cpu.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_bench_only_check
```

### Features-only run

```bash
conda run -n energy-inference python scripts/extract_features.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_features_only_check
```

### Plot a run CSV

```bash
python scripts/plot_results.py \
  --input results/runs/full_cpu_resnet18_resolution_sweep_<run_id>.csv \
  --y latency_ms
```

Useful y-columns:
- `latency_ms`
- `fps`
- `flops_total`
- `flops_total_strict`
- `macs_total`
- `flops_per_sample`

Plot all child runs from one big run directory:

```bash
python scripts/plot_results.py \
  --run-dir results/runs/<big_run_dir> \
  --y latency_ms
```

### Compare FLOPs methods (`fvcore` vs `THOP`)

```bash
python scripts/compare_flops_methods.py \
  --device cpu \
  --models resnet18 resnet50 \
  --batches 1 8 \
  --resolutions 224 640
```

This writes `results/flops/compare_fvcore_vs_thop.csv` with:
- `fvcore_flops` (current project method)
- `thop_macs` (THOP native output)
- `thop_flops_x2` (common alternate FLOPs convention)
- percent-difference columns vs `fvcore_flops`

### Run multiple experiments from a CSV

Use the template:
- `configs/experiments_example.csv`
- Jetson profile: `configs/experiments_jetson_orin.csv`

Run all enabled rows:

```bash
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv
```

When executed, this creates one big run directory under `results/runs/` and writes
one child CSV per enabled row inside it.

Dry run only (validate and preview, no benchmark execution):

```bash
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_example.csv --dry-run
```

Equivalent option on full runner:

```bash
python scripts/run_full.py --experiments-csv configs/experiments_example.csv
```

Dry run through full runner:

```bash
python scripts/run_full.py --experiments-csv configs/experiments_example.csv --dry-run
```

Jetson quick start:

```bash
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv --dry-run
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_jetson_orin.csv
```

CSV notes:
- `enabled`: `1` to run, `0` to skip
- `mode`: `full`, `bench`, or `features`
- list fields are comma-separated (e.g., `1,2,4,8`)
- for `sweep=model`, use `models` list as the true sweep values (`model` can be empty)
- for `sweep=batch`/`sweep=resolution`/`sweep=precision`, `model` is the fixed base model
- leave optional fields empty to use defaults

## 10) FLOPs counting policy

Current project policy is explicit:

- FLOPs/MACs are computed with `THOP` (`ultralytics-thop`).
- `macs_total` stores raw THOP MAC count.
- `flops_total_strict` stores strict FLOPs (`2 * macs_total`).
- `flops_total` is a backward-compatible alias to `flops_total_strict`.
- `unsupported_ops_count` remains in the schema and is forced to `0`.

This is intentional to keep benchmark logs clean and reduce confusion during long sweeps.

## 11) Common mistakes and how to avoid them

- Mistake: appending everything to one file forever  
  - Fix: omit `--out` and let default per-run files be created.

- Mistake: vague or missing experiment tags  
  - Fix: always set `--experiment` with a specific purpose.

- Mistake: forgetting run context  
  - Fix: write concise `--notes` on every run.

- Mistake: changing multiple factors in one run unintentionally  
  - Fix: vary one axis at a time (`model` OR `batch` OR `resolution`).

## 12) Minimal "I forgot everything" restart plan

1. Open `results/run_index.csv`
2. Find the latest `run_id` and `output_path`
3. Read that output CSV
4. Re-run baseline:

```bash
conda run -n energy-inference python scripts/run_full.py \
  --sweep model \
  --models resnet18 resnet50 \
  --experiment cpu_rebaseline \
  --notes "restarting from runbook"
```

