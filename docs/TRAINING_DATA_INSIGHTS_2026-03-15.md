# Training Data Insights (2026-03-15)

This report summarizes the latest analysis output generated from:

- `data/training_data/*.csv` (excluding `*_power_trace.csv`)
- analysis artifacts in `results/plots/training_data_insights_20260315_091055`

## 1) Data Coverage and Quality

- Total rows: **148**
- Successful rows (`status=ok`): **140**
- Failed rows: **8**
- Failure concentration:
  - `vit_b_16`: **47.059%** failure rate
  - all other models: **0%** in this dataset

Source: `summary_overall.csv`, `summary_failure_rate_by_model.csv`

## 2) Classification vs Detection

Median metrics by `model_task`:

- **classification**
  - latency: **24.917 ms**
  - fps: **44.548**
  - total energy: **72.081 J**
  - energy per inference: **55.953 J/inf**
  - power: **15.006 W**
- **detection**
  - latency: **130.776 ms**
  - fps: **18.206**
  - total energy: **426.528 J**
  - energy per inference: **227.277 J/inf**
  - power: **12.016 W**

Interpretation:
- Detection workloads are much heavier per inference than classification in this dataset.
- Classification runs draw higher median power, but complete much faster and therefore consume less energy per inference.

Source: `summary_by_model_task_median.csv`

## 3) Per-Model Highlights

Median values:

- `resnet18`: **10.325 ms**, **98.894 fps**, **31.196 J/inf**, **16.973 W**
- `mobilenet_v3_small`: **22.464 ms**, **44.592 fps**, **54.000 J/inf**, **12.168 W**
- `mobilenet_v3_large`: **26.539 ms**, **37.844 fps**, **65.097 J/inf**, **12.691 W**
- `resnet50`: **24.727 ms**, **40.639 fps**, **73.125 J/inf**, **16.490 W**
- `vit_b_16`: **32.162 ms**, **42.969 fps**, **107.690 J/inf**, **29.520 W**
- `swin_t`: **55.394 ms**, **18.213 fps**, **148.278 J/inf**, **14.680 W**
- `yolo`: **31.873 ms**, **31.515 fps**, **82.111 J/inf**, **13.709 W**
- `ssdlite`: **206.842 ms**, **5.039 fps**, **471.784 J/inf**, **11.831 W**

Interpretation:
- `resnet18` is the strongest efficiency baseline in this dataset.
- `ssdlite` is the most expensive per inference.
- `vit_b_16` shows very high power draw relative to other classification models.

Source: `summary_by_model_median.csv`

## 4) Precision Insights

### 4.1 Raw median by precision

- `fp32`: latency **24.755 ms**, fps **40.713**, energy/inf **77.744 J**, power **15.756 W**
- `fp16`: latency **26.549 ms**, fps **40.734**, energy/inf **65.305 J**, power **14.474 W**
- `bf16`: latency **29.662 ms**, fps **33.911**, energy/inf **85.009 J**, power **13.652 W**

Source: `summary_by_precision_median.csv`

### 4.2 Matched comparison (same model + batch + resolution)

Median deltas vs `fp32`:

- `fp16` vs `fp32`
  - latency: **+17.625%**
  - power: **-10.315%**
  - energy per inference: **+3.883%**
- `bf16` vs `fp32`
  - latency: **+22.079%**
  - power: **-10.808%**
  - energy per inference: **+9.470%**

Interpretation:
- Lower precision reduces instantaneous power, but in this dataset it increases latency enough to worsen energy per inference.
- For matched points, `fp32` is best on latency and energy per inference.

Source: `summary_precision_deltas.csv`

## 5) Recommended Reporting Statements

- “Across matched configurations, `fp16`/`bf16` reduced power but increased latency, resulting in higher median energy per inference than `fp32`.”
- “Detection workloads are substantially more energy-intensive per inference than classification workloads in the current benchmark set.”
- “`resnet18` provides the best efficiency baseline among tested classification models, while `ssdlite` is the most expensive per inference.”

## 6) Key Figures

### Task-level comparisons

![Latency by model task](../results/plots/training_data_insights_20260315_091055/bar_latency_ms_by_model_task.png)

![Power by model task](../results/plots/training_data_insights_20260315_091055/bar_power_total_W_by_model_task.png)

### Precision behavior

![Latency vs precision](../results/plots/training_data_insights_20260315_091055/line_latency_ms_vs_precision.png)

![Power vs precision](../results/plots/training_data_insights_20260315_091055/line_power_total_W_vs_precision.png)

![Energy per inference vs precision](../results/plots/training_data_insights_20260315_091055/line_energy_per_inf_J_vs_precision.png)

### Model spread and correlation

![Energy per inference by model](../results/plots/training_data_insights_20260315_091055/box_energy_per_inf_J_by_model.png)

![Correlation heatmap (ok rows)](../results/plots/training_data_insights_20260315_091055/corr_heatmap_ok_rows.png)

## 7) Caveats

- `vit_b_16` resolution sweeps include expected failures for non-224 input sizes; this affects representativeness.
- Global correlation/medians can mix different regimes; matched comparisons and per-task analysis are preferred for causal interpretation.
