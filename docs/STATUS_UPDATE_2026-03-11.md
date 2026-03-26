# Project Status Update (2026-03-11)

## Executive Summary

The benchmarking pipeline has been significantly expanded beyond the initial ResNet-only setup.  
We now support multiple model families, energy/power logging, precision sweeps, and a stronger energy-prediction baseline with task-specific models.

Core benchmarking is functional and producing useful results for classification models. Detection-model prediction quality is currently weaker and requires additional data/model specialization.

## What Is Implemented

- Unified runner renamed to `scripts/run_full.py` (replacing `run_full_cpu.py`).
- Supported benchmark models now include:
  - `resnet18`, `resnet50`
  - `mobilenet_v3_large`, `mobilenet_v3_small`
  - `vit_b_16`, `swin_t`
  - `ssdlite320_mobilenet_v3_large` (alias: `ssdlite`)
  - `yolov8n` (alias: `yolo`, requires `ultralytics`)
- Added task tagging in outputs:
  - `model_task=classification|detection`
- Added power metric in run CSVs:
  - `power_total_W` (average total power over benchmark window)
- FLOPs/MAC accounting upgraded:
  - THOP backend in use
  - `macs_total`
  - `flops_total_strict` (`2 * macs_total`)
  - `flops_total` retained as compatibility alias to strict FLOPs
- Added precision sweep support end-to-end:
  - sweep axis now supports `precision`
  - precision options currently used: `fp32`, `fp16`, `bf16`
  - precision is actually applied during benchmarking via autocast
- Added comprehensive benchmark config:
  - `configs/experiments_comprehensive.csv`
  - covers model/batch/resolution/precision across all currently supported models

## Energy Predictor Status

Training/prediction pipeline has evolved from a single global linear regressor to a stronger approach:

- Features now include:
  - numeric: `flops_total`, `batch`, `resolution`
  - categorical one-hot: `model`, `precision`
- Regressor:
  - `HistGradientBoostingRegressor`
  - trained on `log1p(energy_cpu_J)` target, inverted with `expm1` at inference
  - inference output clipped to `>= 0` (prevents physically invalid negative energy)
- Models are now trained separately by task:
  - `classification`
  - `detection`

Latest reported metrics:

- classification: `rows=108`, `R^2=0.3993`, `MAE=3.043653 J`
- detection: `rows=32`, `R^2=-0.0846`, `MAE=36.674060 J`

Interpretation:
- Classification predictor is usable as a baseline.
- Detection predictor is currently weak (insufficient data and likely mixed regime complexity).

## Known Issues / Caveats

- ViT resolution constraints:
  - `vit_b_16` is fixed-size (224), so non-224 resolution sweeps fail by design.
- THOP hook contamination issue observed in some `full` runs:
  - error example: `'Conv2d' object has no attribute 'total_ops'`
  - planned fix: run FLOPs profiling on a temporary model instance (not the benchmark model).
- Training-data hygiene:
  - `scripts/train_energy_model.py` reads all `*.csv` in `data/training_data/`; avoid placing raw `*_power_trace.csv` files there.

## Immediate Next Steps

1. Implement THOP isolation fix in `full` mode for robust transformer/detector runs.
2. Improve detection predictor by splitting further (`ssdlite` vs `yolo`) and collecting more detection samples.
3. Add explicit training-data filtering safeguards:
   - skip files without required schema
   - explicitly filter to `status=="ok"` where available.
4. Regenerate comprehensive runs after THOP isolation for cleaner training data.

## Repro Commands (Current)

```bash
# Comprehensive benchmark plan
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_comprehensive.csv --dry-run
python scripts/run_experiments_csv.py --experiments-csv configs/experiments_comprehensive.csv

# Train energy predictor
python scripts/train_energy_model.py

# Predict one point
python scripts/predict_energy.py \
  --flops-total 3.8004576256e10 \
  --model resnet18 \
  --batch 1 \
  --resolution 640 \
  --precision fp16
```
