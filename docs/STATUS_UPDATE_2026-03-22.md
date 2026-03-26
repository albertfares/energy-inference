# Project Status Update (2026-03-22)

## Executive Summary

Since the 2026-03-11 update, the project moved from a mostly task-level benchmarking/prediction workflow to a more robust "data curation + per-group modeling + interactive testing" workflow.

The benchmark scope now includes additional classification families (`googlenet`, `shufflenet_v2_x1_0`, `vgg16`), and the training pipeline now supports training separate regressors per model (not only per task). We also added utility scripts for filtering large cartesian runs into clean training subsets and for interactive predictor testing.

## What Changed Since 2026-03-11

- Expanded default cartesian benchmark model set in `scripts/run_full_cartesian.py`:
  - added `googlenet`, `shufflenet_v2_x1_0`, `vgg16`
- Extended model factory support in `src/energy_inference/models.py`:
  - added `googlenet` (`aux_logits=False`)
  - added `shufflenet_v2_x1_0`
  - added `vgg16`
  - aliases supported: `vdd -> vgg16`, `shufflenet -> shufflenet_v2_x1_0`
- Improved model-family tagging in `src/energy_inference/features.py`:
  - `googlenet/inception -> inception`
  - `shufflenet -> shufflenet`
  - `vgg/vdd -> vgg`
- Added cartesian-run filtering utility:
  - `scripts/filter_cartesian_run.py`
  - keeps only rows matching target status and selected batches (default: `status=ok`, `batch in {1,2,4}`)
- Upgraded training pipeline in `scripts/train_energy_model.py`:
  - `--include-model-feature` to control model one-hot feature usage
  - `--separate-by {model_task,model}` to train one regressor per task or per model
  - persisted payload now includes grouping metadata (`group_by`, `models_by_group`, `feature_names_by_group`)
- Updated inference loader in `scripts/predict_energy.py`:
  - supports new grouped payload format (`group_by`)
  - remains backward compatible with legacy task-grouped payloads
- Added interactive predictor test pipeline:
  - `scripts/test_predictor_pipeline.py`
  - prompts one-by-one for model, FLOPs, batch, resolution, precision, optional task override
- Added/expanded live camera detection and streaming workflow:
  - `scripts/live_detect_ssdlite.py` supports multiple detection backends and RTP/MJPEG streaming + optional live INA3221 summaries
  - operational runbook in `docs/CAMERA_STREAMING_JETSON_TO_MAC.md`

## Current Data Snapshot (As Of This Update)

- Latest comprehensive cartesian run file:
  - `results/runs/full_cartesian_comprehensive_cartesian_20260317_145855_989.csv`
  - rows: **1532**
  - `status=ok`: **1370**
  - `status=failed`: **162**
- Failure concentration remains dominated by `vit_b_16` non-supported input sizes:
  - `vit_b_16` ok-rate in this run: **0.100**
  - all other currently included models in this run: **1.000** ok-rate
- Curated filtered training dataset:
  - `data/training_data/filtred_data.csv`
  - rows: **909**
  - all rows `status=ok`
  - model coverage: **11 models** (`resnet18`, `resnet50`, `mobilenet_v3_large`, `mobilenet_v3_small`, `googlenet`, `shufflenet_v2_x1_0`, `vgg16`, `vit_b_16`, `swin_t`, `ssdlite`, `yolo`)

## Predictor Workflow Status

The predictor path now supports two clear modes:

1. **Per-task regressors** (`--separate-by model_task`)
   - suitable baseline when per-model data is limited.
2. **Per-model regressors** (`--separate-by model`)
   - better aligned with model-specific behavior and non-linear regime differences.

Inference now auto-selects the right regressor based on saved grouping metadata and user-provided `--model` (or task override).

## Known Issues / Caveats

- `vit_b_16` remains fixed-resolution oriented; broad resolution sweeps still produce expected failures.
- Large cartesian outputs can mix very different operating regimes; filtered training subsets are now preferred before model fitting.
- Local tool/runtime dependencies still need to be present per machine (`torch`, `joblib`, etc.) for full script execution.

## Recommended Next Steps

1. Train and compare two predictor artifacts on the same curated dataset:
   - `--separate-by model_task`
   - `--separate-by model`
2. Track per-model MAE/R2 in a small results table and decide default deployment mode.
3. Split out `vit_b_16` resolution handling in benchmark configs (or constrain to valid sizes) to reduce noisy failure-heavy rows.
4. Add a lightweight evaluation script that replays a holdout CSV and reports per-group error in one command.

## Repro Commands (Current)

```bash
# 1) Filter a cartesian run for clean training rows
python scripts/filter_cartesian_run.py \
  --input results/runs/full_cartesian_comprehensive_cartesian_20260317_145855_989.csv \
  --output data/training_data/filtred_data.csv \
  --status ok \
  --batches 1 2 4

# 2) Train grouped predictor (per model)
python scripts/train_energy_model.py --separate-by model

# 3) Predict one point from CLI
python scripts/predict_energy.py \
  --flops-total 3.8004576256e10 \
  --model resnet18 \
  --batch 1 \
  --resolution 640 \
  --precision fp16

# 4) Interactive predictor testing loop
python scripts/test_predictor_pipeline.py
```
