"""
Leave-one-config-out cross-validation for the decomposed energy predictor.

For each unique (model, imgsz, precision) configuration we:
  1. Retrain ALL stage sub-predictors from scratch on the remaining configs.
  2. Re-estimate P_overhead from the remaining configs.
  3. Run the full predict_pipeline on every row of the held-out config.
  4. Record per-stage latency + energy errors AND end-to-end FPS + energy errors.

This gives an honest estimate of how well the predictor generalises to a
configuration it has never seen — the key question for deployment.

Outputs
-------
  results/analysis/loocv_decomposed_<ts>/
    loocv_detailed.csv          — one row per held-out sweep run
    loocv_summary_by_config.csv — per held-out config: FPS/E MAPE, bias, fragile stage
    loocv_stage_errors.csv      — stage × metric MAPE/bias aggregated over all folds
    fragility_heatmap.png       — stage × config energy-error heatmap
    loocv_pred_vs_actual.png    — FPS + energy scatter for LOOCV predictions
    loocv_error_by_fps.png      — MAPE vs target_fps, split by config
    loocv_config_summary_bar.png— per-config MAPE bars (FPS + energy)
    loocv_bias_chart.png        — signed bias per config (over vs under prediction)

Usage
-----
    python3 scripts/loocv_analysis_decomposed_predictor.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import training module so joblib can unpickle its objects, and to reuse helpers
import train_decomposed_predictor as TDP
from train_decomposed_predictor import (
    load_data,
    config_key,
    capture_features,
    preprocess_features,
    infer_features,
    postprocess_features,
    estimate_overhead,
    estimate_camera_constants,
    compute_capture_outputs,
    MODEL_LABELS,
    MODEL_FAMILY,
    STAGES,
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = PROJECT_ROOT / "results" / "analysis" / f"loocv_decomposed_{TIMESTAMP}"

STAGE_COLORS = {
    "capture":     "#4C72B0",
    "preprocess":  "#DD8452",
    "infer":       "#55A868",
    "postprocess": "#C44E52",
    "overhead":    "#8172B2",
}

STAGE_COL_MAP = {
    "capture":    ("capture_lat_ms",    "capture_e_mj"),
    "preprocess": ("preprocess_lat_ms", "preprocess_e_mj"),
    "infer":      ("infer_lat_ms",      "infer_energy_mj"),
    "postprocess":("postprocess_lat_ms","postprocess_energy_mj"),
}

FEATURE_FNS = {
    "capture":    capture_features,
    "preprocess": preprocess_features,
    "infer":      infer_features,
    "postprocess":postprocess_features,
}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge",  Ridge(alpha=1.0)),
    ])


def safe_mape(actual: np.ndarray, pred: np.ndarray, thr: float = 1e-3) -> float:
    mask = np.abs(actual) > thr
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / np.abs(actual[mask])) * 100)


def safe_bias(actual: np.ndarray, pred: np.ndarray, thr: float = 1e-3) -> float:
    """Mean signed relative error (%); positive = over-predict."""
    mask = np.abs(actual) > thr
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((pred[mask] - actual[mask]) / np.abs(actual[mask])) * 100)


def train_stage_on_subset(df: pd.DataFrame, stage: str) -> dict:
    """Train latency + energy models for one stage on the given data subset."""
    feat_fn   = FEATURE_FNS[stage]
    lat_col, energy_col = STAGE_COL_MAP[stage]
    X = feat_fn(df)
    models: dict[str, object] = {}
    for col in [lat_col, energy_col]:
        y = df[col]
        if y.isna().any():
            models[col] = None
        elif y.std() < 1e-9:
            models[col] = "zero"
        else:
            m = make_model()
            m.fit(X, y)
            models[col] = m
    return models


def predict_with_local_payload(
    payload: dict,
    model_short: str,
    imgsz: int,
    precision: str,
    target_fps: float,
    width: int,
    height: int,
) -> dict:
    """
    Run the combination layer using a locally built payload dict.
    Mirrors predict_pipeline() from train_decomposed_predictor but accepts
    a payload built inside this script (no joblib required).
    """
    is_fp16 = float(precision.lower() == "fp16")
    family  = MODEL_FAMILY.get(model_short, "unknown")
    eff_fps = float(target_fps) if target_fps > 0 else 30.0

    row = pd.DataFrame([{
        "model_short":  model_short,
        "model_family": family,
        "is_fp16":      is_fp16,
        "imgsz":        float(imgsz),
        "width":        float(width),
        "height":       float(height),
        "target_fps":   float(target_fps),
        "eff_fps":      eff_fps,
    }])

    results: dict[str, float] = {}
    for stage in STAGES:
        feat_fn          = FEATURE_FNS[stage]
        lat_col, e_col   = STAGE_COL_MAP[stage]
        s_models         = payload["stage_models"][stage]
        X_pred           = feat_fn(row)

        for col_key, out_key in [(lat_col, f"T_{stage}_ms"), (e_col, f"E_{stage}_mj")]:
            m = s_models.get(col_key)
            if m is None or m == "zero":
                results[out_key] = 0.0
            else:
                results[out_key] = float(np.maximum(m.predict(X_pred)[0], 0.0))

    p_overhead = payload["p_overhead_w"]

    # Phase 2: override capture from the combination layer using camera constants
    T_other_ms = (results["T_preprocess_ms"] + results["T_infer_ms"] +
                  results["T_postprocess_ms"])
    camera_constants = payload.get("camera_constants")
    if camera_constants:
        T_cap, E_cap = compute_capture_outputs(
            camera_constants, p_overhead, width, height, target_fps, T_other_ms
        )
        results["T_capture_ms"] = T_cap
        results["E_capture_mj"] = E_cap

    T_compute = (results["T_capture_ms"] + results["T_preprocess_ms"] +
                 results["T_infer_ms"]   + results["T_postprocess_ms"])

    T_frame = max(T_compute, 1000.0 / target_fps) if target_fps > 0 else T_compute
    fps     = 1000.0 / max(T_frame, 1e-3)

    # Sleep-only overhead (Phase 1)
    T_sleep_ms = max(0.0, T_frame - T_compute)
    E_overhead = p_overhead * (T_sleep_ms / 1000.0) * 1000   # mJ
    E_total    = (results["E_capture_mj"] + results["E_preprocess_mj"] +
                  results["E_infer_mj"]   + results["E_postprocess_mj"] + E_overhead)

    return {
        "fps":              round(fps, 2),
        "T_frame_ms":       round(T_frame, 2),
        "E_total_mj":       round(E_total, 1),
        "T_capture_ms":     round(results["T_capture_ms"],    2),
        "E_capture_mj":     round(results["E_capture_mj"],    1),
        "T_preprocess_ms":  round(results["T_preprocess_ms"], 2),
        "E_preprocess_mj":  round(results["E_preprocess_mj"], 1),
        "T_infer_ms":       round(results["T_infer_ms"],      2),
        "E_infer_mj":       round(results["E_infer_mj"],      1),
        "T_postprocess_ms": round(results["T_postprocess_ms"],2),
        "E_postprocess_mj": round(results["E_postprocess_mj"],1),
        "E_overhead_mj":    round(E_overhead, 1),
        "bottleneck":       "throttle" if T_frame > T_compute else "compute",
    }


# ── Core LOOCV loop ────────────────────────────────────────────────────────────

def run_loocv(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique (model, imgsz, precision) config:
      - Build train split = all other configs
      - Retrain all stage models + overhead on train split
      - Predict every row of the held-out config
      - Record predicted vs actual for all stages + end-to-end
    """
    configs     = config_key(df)
    unique_cfgs = sorted(configs.unique())
    n_cfgs      = len(unique_cfgs)

    all_rows = []

    for fold_i, hold in enumerate(unique_cfgs):
        test_mask  = configs == hold
        train_mask = ~test_mask
        df_train   = df[train_mask].copy()
        df_test    = df[test_mask].copy()

        n_train = train_mask.sum()
        n_test  = test_mask.sum()
        print(f"\n[{fold_i+1}/{n_cfgs}] Hold out: {hold}  "
              f"(train n={n_train}, test n={n_test})")

        if n_train < 5:
            print(f"  SKIP — only {n_train} training rows, not enough to fit.")
            continue

        # ── Retrain every stage ────────────────────────────────────────────────
        stage_models: dict[str, dict] = {}
        for stage in STAGES:
            stage_models[stage] = train_stage_on_subset(df_train, stage)

        # ── Re-estimate overhead and camera constants from training subset ────
        # (Both are derived per fold so the held-out config never leaks in.)
        p_overhead       = estimate_overhead(df_train)
        camera_constants = estimate_camera_constants(df_train)

        local_payload = {
            "stage_models":     stage_models,
            "p_overhead_w":     p_overhead,
            "camera_constants": camera_constants,
        }

        # ── Predict held-out rows ──────────────────────────────────────────────
        for _, row in df_test.iterrows():
            pred = predict_with_local_payload(
                local_payload,
                model_short = row["model_short"],
                imgsz       = int(row["imgsz"]),
                precision   = row["precision"],
                target_fps  = float(row["target_fps"]),
                width       = int(row["width"]),
                height      = int(row["height"]),
            )

            actual_T_frame = (row["capture_lat_ms"] + row["preprocess_lat_ms"] +
                              row["infer_lat_ms"]   + row["postprocess_lat_ms"])

            r: dict = {
                # Identity
                "held_out_config": hold,
                "model":      row["model_short"],
                "imgsz":      int(row["imgsz"]),
                "precision":  row["precision"],
                "target_fps": row["target_fps"],

                # Predicted stage latencies (ms)
                "pred_T_capture_ms":    pred["T_capture_ms"],
                "pred_T_preprocess_ms": pred["T_preprocess_ms"],
                "pred_T_infer_ms":      pred["T_infer_ms"],
                "pred_T_postprocess_ms":pred["T_postprocess_ms"],
                "pred_T_frame_ms":      pred["T_frame_ms"],

                # Actual stage latencies (ms)
                "actual_T_capture_ms":    row["capture_lat_ms"],
                "actual_T_preprocess_ms": row["preprocess_lat_ms"],
                "actual_T_infer_ms":      row["infer_lat_ms"],
                "actual_T_postprocess_ms":row["postprocess_lat_ms"],
                "actual_T_frame_ms":      actual_T_frame,

                # Predicted stage energies (mJ)
                "pred_E_capture_mj":    pred["E_capture_mj"],
                "pred_E_preprocess_mj": pred["E_preprocess_mj"],
                "pred_E_infer_mj":      pred["E_infer_mj"],
                "pred_E_postprocess_mj":pred["E_postprocess_mj"],
                "pred_E_overhead_mj":   pred["E_overhead_mj"],
                "pred_E_total_mj":      pred["E_total_mj"],

                # Actual stage energies (mJ)
                "actual_E_capture_mj":    row["capture_e_mj"],
                "actual_E_preprocess_mj": row["preprocess_e_mj"],
                "actual_E_infer_mj":      row["infer_energy_mj"],
                "actual_E_postprocess_mj":row["postprocess_energy_mj"],
                "actual_E_idle_mj":       row["idle_mj"],
                "actual_E_total_mj":      row["energy_per_frame_mj"],

                # End-to-end
                "pred_fps":   pred["fps"],
                "actual_fps": row["fps_mean"],
                "bottleneck": pred["bottleneck"],
                "p_overhead_w": p_overhead,
            }

            # ── Per-stage signed relative errors (%) ──────────────────────────
            def rel_err(p_val: float, a_val: float) -> float:
                return (p_val - a_val) / max(abs(a_val), 1e-3) * 100 \
                    if abs(a_val) > 1e-3 else float("nan")

            for stage, (t_pred_k, e_pred_k), (t_act_k, e_act_k) in [
                ("capture",
                 ("T_capture_ms",    "E_capture_mj"),
                 ("capture_lat_ms",  "capture_e_mj")),
                ("preprocess",
                 ("T_preprocess_ms", "E_preprocess_mj"),
                 ("preprocess_lat_ms","preprocess_e_mj")),
                ("infer",
                 ("T_infer_ms",       "E_infer_mj"),
                 ("infer_lat_ms",     "infer_energy_mj")),
                ("postprocess",
                 ("T_postprocess_ms", "E_postprocess_mj"),
                 ("postprocess_lat_ms","postprocess_energy_mj")),
            ]:
                r[f"{stage}_T_err_pct"] = rel_err(pred[t_pred_k], row[t_act_k])
                r[f"{stage}_E_err_pct"] = rel_err(pred[e_pred_k], row[e_act_k])

            # End-to-end errors
            r["fps_err_pct"]    = rel_err(pred["fps"],        row["fps_mean"])
            r["E_total_err_pct"]= rel_err(pred["E_total_mj"], row["energy_per_frame_mj"])

            all_rows.append(r)

    return pd.DataFrame(all_rows)


# ── Summaries ──────────────────────────────────────────────────────────────────

def summary_by_config(results: pd.DataFrame) -> pd.DataFrame:
    """Per held-out config: MAPE + bias + fragile stage."""
    rows = []
    for cfg, grp in results.groupby("held_out_config"):
        row: dict = {
            "config":              cfg,
            "n_rows":              len(grp),
            "fps_MAPE_%":          safe_mape(grp["actual_fps"].values,       grp["pred_fps"].values),
            "fps_bias_%":          safe_bias(grp["actual_fps"].values,       grp["pred_fps"].values),
            "E_total_MAPE_%":      safe_mape(grp["actual_E_total_mj"].values,grp["pred_E_total_mj"].values),
            "E_total_bias_%":      safe_bias(grp["actual_E_total_mj"].values,grp["pred_E_total_mj"].values),
        }
        # Per-stage energy MAPE
        stage_mapes: dict[str, float] = {}
        for stage in STAGES:
            col = f"{stage}_E_err_pct"
            vals = grp[col].dropna()
            mape = np.mean(np.abs(vals)) if len(vals) else float("nan")
            row[f"{stage}_E_MAPE_%"] = round(mape, 1)
            stage_mapes[stage] = mape if not np.isnan(mape) else -1.0

        # Fragile stage = highest energy MAPE (skip zero stages)
        valid_stages = {s: v for s, v in stage_mapes.items() if v > 0}
        row["fragile_stage"] = max(valid_stages, key=valid_stages.get) if valid_stages else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def stage_error_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per stage × metric: MAPE, bias, std over all LOOCV rows."""
    rows = []
    for stage in STAGES:
        for metric, label in [("T", "Latency (ms)"), ("E", "Energy (mJ)")]:
            col  = f"{stage}_{metric}_err_pct"
            vals = results[col].dropna()
            rows.append({
                "stage":    stage,
                "metric":   label,
                "MAPE_%":   round(np.mean(np.abs(vals)), 1) if len(vals) else float("nan"),
                "bias_%":   round(np.mean(vals),         1) if len(vals) else float("nan"),
                "std_%":    round(np.std(vals),          1) if len(vals) else float("nan"),
                "n_valid":  int(len(vals)),
            })
    # Add end-to-end rows
    for col, label in [("fps_err_pct", "FPS"), ("E_total_err_pct", "Total energy (mJ)")]:
        vals = results[col].dropna()
        rows.append({
            "stage":   "end-to-end",
            "metric":  label,
            "MAPE_%":  round(np.mean(np.abs(vals)), 1) if len(vals) else float("nan"),
            "bias_%":  round(np.mean(vals),         1) if len(vals) else float("nan"),
            "std_%":   round(np.std(vals),          1) if len(vals) else float("nan"),
            "n_valid": int(len(vals)),
        })
    return pd.DataFrame(rows)


# ── Insights printer ───────────────────────────────────────────────────────────

def print_insights(cfg_summary: pd.DataFrame, stage_summary: pd.DataFrame) -> None:
    print("\n" + "═"*70)
    print("  LOOCV INSIGHTS — Fragility Analysis")
    print("═"*70)

    print("\n  Per held-out config:")
    print(f"  {'Config':<42}  {'FPS MAPE':>9}  {'E MAPE':>8}  {'Bias E%':>8}  {'Fragile stage'}")
    print(f"  {'─'*42}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*14}")
    for _, r in cfg_summary.iterrows():
        fps_m = f"{r['fps_MAPE_%']:.1f}%" if not np.isnan(r['fps_MAPE_%']) else "  n/a"
        e_m   = f"{r['E_total_MAPE_%']:.1f}%" if not np.isnan(r['E_total_MAPE_%']) else "  n/a"
        e_b   = f"{r['E_total_bias_%']:+.1f}%" if not np.isnan(r['E_total_bias_%']) else "  n/a"
        print(f"  {r['config']:<42}  {fps_m:>9}  {e_m:>8}  {e_b:>8}  {r['fragile_stage']}")

    print("\n  Overall stage accuracy (LOOCV):")
    print(f"  {'Stage':<14}  {'Metric':<18}  {'MAPE':>7}  {'Bias':>8}  {'Std':>7}  {'n'}")
    print(f"  {'─'*14}  {'─'*18}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*5}")
    for _, r in stage_summary.iterrows():
        mape = f"{r['MAPE_%']:.1f}%" if not np.isnan(r['MAPE_%']) else "  n/a"
        bias = f"{r['bias_%']:+.1f}%" if not np.isnan(r['bias_%']) else "  n/a"
        std  = f"{r['std_%']:.1f}%"  if not np.isnan(r['std_%'])  else "  n/a"
        print(f"  {r['stage']:<14}  {r['metric']:<18}  {mape:>7}  {bias:>8}  {std:>7}  {r['n_valid']}")

    print("\n  Key observations:")

    # Find most fragile config
    valid = cfg_summary.dropna(subset=["E_total_MAPE_%"])
    if not valid.empty:
        worst = valid.loc[valid["E_total_MAPE_%"].idxmax()]
        best  = valid.loc[valid["E_total_MAPE_%"].idxmin()]
        print(f"  ✗ Most fragile config  : {worst['config']}  "
              f"(E MAPE={worst['E_total_MAPE_%']:.1f}%,  fragile stage={worst['fragile_stage']})")
        print(f"  ✓ Best generalising    : {best['config']}  "
              f"(E MAPE={best['E_total_MAPE_%']:.1f}%)")

    # Systematic bias
    biased = cfg_summary.dropna(subset=["E_total_bias_%"])
    over   = biased[biased["E_total_bias_%"] >  15]
    under  = biased[biased["E_total_bias_%"] < -15]
    if not over.empty:
        print(f"  ↑ Over-predicted energy : {', '.join(over['config'].tolist())}  "
              f"(bias > +15%)")
    if not under.empty:
        print(f"  ↓ Under-predicted energy: {', '.join(under['config'].tolist())}  "
              f"(bias < -15%)")

    # Stage-level warnings
    stage_e = stage_summary[stage_summary["metric"] == "Energy (mJ)"].set_index("stage")
    for stage in STAGES:
        if stage not in stage_e.index:
            continue
        m = stage_e.loc[stage, "MAPE_%"]
        if not np.isnan(m) and m > 20:
            print(f"  ⚠ Stage '{stage}' energy MAPE={m:.1f}% — "
                  f"high generalisation error, may need more diverse training data.")

    print("═"*70 + "\n")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_fragility_heatmap(results: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: stage × held-out-config, cell = mean |E error %|."""
    configs = sorted(results["held_out_config"].unique())
    stage_e_cols = [
        ("capture",    "capture_E_err_pct"),
        ("preprocess", "preprocess_E_err_pct"),
        ("infer",      "infer_E_err_pct"),
        ("postprocess","postprocess_E_err_pct"),
        ("end-to-end", "E_total_err_pct"),
    ]

    mat = np.full((len(stage_e_cols), len(configs)), float("nan"))
    for j, cfg in enumerate(configs):
        sub = results[results["held_out_config"] == cfg]
        for i, (_, col) in enumerate(stage_e_cols):
            vals = sub[col].dropna()
            if len(vals) > 0:
                mat[i, j] = np.mean(np.abs(vals))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 2.0), 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=30)
    plt.colorbar(im, ax=ax, label="Mean |error| (%)")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(stage_e_cols)))
    ax.set_yticklabels([s for s, _ in stage_e_cols])
    ax.set_title("LOOCV Fragility — Stage Energy Error per Held-out Config\n"
                 "(green = generalises well, red = fragile)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if v > 20 else "black")

    fig.tight_layout()
    p = out_dir / "fragility_heatmap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_pred_vs_actual(results: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: LOOCV predicted vs actual for FPS and total energy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LOOCV — Predicted vs Actual (held-out configs)", fontsize=12)

    colors  = plt.cm.tab10.colors
    configs = sorted(results["held_out_config"].unique())
    cmap    = {c: colors[i % len(colors)] for i, c in enumerate(configs)}

    for ax, pred_col, act_col, label, unit in [
        (axes[0], "pred_fps",        "actual_fps",        "FPS",          "fps"),
        (axes[1], "pred_E_total_mj", "actual_E_total_mj", "Energy/frame", "mJ"),
    ]:
        for cfg in configs:
            sub = results[results["held_out_config"] == cfg]
            ax.scatter(sub[act_col], sub[pred_col],
                       label=cfg, color=cmap[cfg], alpha=0.75, s=35,
                       edgecolors="none")

        lo = min(results[act_col].min(), results[pred_col].min()) * 0.9
        hi = max(results[act_col].max(), results[pred_col].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")

        mape = safe_mape(results[act_col].values, results[pred_col].values)
        mae  = float(np.mean(np.abs(results[pred_col].values - results[act_col].values)))
        ax.set_title(f"{label} (LOOCV)\nMAPE={mape:.1f}%  MAE={mae:.2f} {unit}")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p = out_dir / "loocv_pred_vs_actual.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_error_by_fps(results: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: MAPE vs target_fps, split by held-out config."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LOOCV Error vs Target FPS", fontsize=12)

    colors  = plt.cm.tab10.colors
    configs = sorted(results["held_out_config"].unique())
    fps_vals= sorted(results["target_fps"].unique())

    for ax, err_col, ylabel in [
        (axes[0], "fps_err_pct",    "FPS |error| (%)"),
        (axes[1], "E_total_err_pct","Energy/frame |error| (%)"),
    ]:
        for i, cfg in enumerate(configs):
            sub = results[results["held_out_config"] == cfg]
            mapes = []
            for fps in fps_vals:
                vals = sub[sub["target_fps"] == fps][err_col].dropna()
                mapes.append(np.mean(np.abs(vals)) if len(vals) else float("nan"))
            ax.plot(fps_vals, mapes, marker="o", label=cfg,
                    color=colors[i % len(colors)], linewidth=1.8)

        ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="10% threshold")
        ax.set_xlabel("Target FPS (0 = unbounded)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)

    fig.tight_layout()
    p = out_dir / "loocv_error_by_fps.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_config_summary_bars(cfg_summary: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bar chart: per-config FPS MAPE and energy MAPE side by side.
    Immediately shows which configs generalise best/worst.
    """
    cfgs   = cfg_summary["config"].tolist()
    x      = np.arange(len(cfgs))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(cfgs) * 1.8), 5))
    bars_fps = ax.bar(x - width/2, cfg_summary["fps_MAPE_%"].fillna(0),
                      width, label="FPS MAPE",    color="#4C72B0", alpha=0.85)
    bars_e   = ax.bar(x + width/2, cfg_summary["E_total_MAPE_%"].fillna(0),
                      width, label="Energy MAPE", color="#DD8452", alpha=0.85)

    ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="10% target")
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("LOOCV Accuracy per Held-out Config\n"
                 "(lower = better generalisation to unseen config)")
    ax.legend()

    for bar in list(bars_fps) + list(bars_e):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    p = out_dir / "loocv_config_summary_bar.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_bias_chart(cfg_summary: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal bar chart: signed energy bias per config.
    Positive = predictor over-estimates, negative = under-estimates.
    """
    cfgs = cfg_summary["config"].tolist()
    bias = cfg_summary["E_total_bias_%"].fillna(0).tolist()

    colors = ["#DD8452" if b > 0 else "#4C72B0" for b in bias]
    fig, ax = plt.subplots(figsize=(7, max(4, len(cfgs) * 0.55)))
    y = np.arange(len(cfgs))
    ax.barh(y, bias, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 10, color="gray", linestyle="--", linewidth=1)
    ax.axvline(-10, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(cfgs, fontsize=9)
    ax.set_xlabel("Mean signed error (%)  +over-predict / −under-predict")
    ax.set_title("LOOCV Energy Prediction Bias per Config")

    for i, (b, bar_y) in enumerate(zip(bias, y)):
        ax.text(b + (0.5 if b >= 0 else -0.5), bar_y,
                f"{b:+.1f}%", va="center",
                ha="left" if b >= 0 else "right", fontsize=8)

    fig.tight_layout()
    p = out_dir / "loocv_bias_chart.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}\n")

    df = load_data()
    configs = config_key(df)
    print(f"\nConfigs in dataset ({configs.nunique()} unique):")
    for cfg, grp in df.groupby(configs):
        print(f"  {cfg:<45s}  n={len(grp)}")

    # ── Run LOOCV ──────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("Running leave-one-config-out cross-validation …")
    print("─"*70)
    results = run_loocv(df)

    if results.empty:
        print("ERROR: no LOOCV results produced. Check that the dataset has "
              "at least 2 distinct configs.")
        sys.exit(1)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    results.to_csv(OUT_DIR / "loocv_detailed.csv", index=False)
    print(f"\nSaved loocv_detailed.csv  ({len(results)} rows)")

    cfg_summary   = summary_by_config(results)
    stage_summary = stage_error_summary(results)

    cfg_summary.to_csv(OUT_DIR / "loocv_summary_by_config.csv", index=False)
    stage_summary.to_csv(OUT_DIR / "loocv_stage_errors.csv",    index=False)
    print("Saved loocv_summary_by_config.csv and loocv_stage_errors.csv")

    # ── Print insights ─────────────────────────────────────────────────────────
    print_insights(cfg_summary, stage_summary)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("── Generating plots ────────────────────────────────────────")
    plot_fragility_heatmap(results, OUT_DIR)
    plot_pred_vs_actual(results, OUT_DIR)
    plot_error_by_fps(results, OUT_DIR)
    plot_config_summary_bars(cfg_summary, OUT_DIR)
    plot_bias_chart(cfg_summary, OUT_DIR)

    print(f"\nDone. All outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
