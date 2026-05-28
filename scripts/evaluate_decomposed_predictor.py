"""
Comprehensive in-sample evaluation of the decomposed energy predictor.

For every row in the sweep data, runs the predictor and compares predicted
vs actual values at the stage level (latency + energy) and end-to-end
(FPS + total energy).

Outputs
-------
  results/analysis/eval_decomposed_<ts>/
    detailed_results.csv   — one row per sweep run, all predicted + actual values
    summary_by_config.csv  — mean absolute error per (model, imgsz, precision)
    summary_by_fps.csv     — mean absolute error per target_fps
    stage_errors.csv       — per-stage MAPE/MAE/bias aggregated
    stage_breakdown.png    — predicted vs actual stage breakdown bars
    error_by_fps.png       — end-to-end error as a function of target FPS
    stage_error_heatmap.png— stage × config MAPE heatmap
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Must import training module before joblib.load (pickle references)
import train_decomposed_predictor as TDP
from train_decomposed_predictor import (
    load_data, config_key, predict_pipeline,
    STAGE_CFG, STAGES,
)
import joblib

PREDICTOR_PATH = PROJECT_ROOT / "models" / "decomposed_predictor.pkl"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = PROJECT_ROOT / "results" / "analysis" / f"eval_decomposed_{TIMESTAMP}"

STAGE_COLORS = {
    "capture":     "#4C72B0",
    "preprocess":  "#DD8452",
    "infer":       "#55A868",
    "postprocess": "#C44E52",
    "overhead":    "#8172B2",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def safe_mape(actual: np.ndarray, pred: np.ndarray, threshold: float = 1e-3) -> float:
    mask = np.abs(actual) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / np.abs(actual[mask])) * 100)


def safe_bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """Signed mean error as % of actual — positive = over-predict."""
    mask = np.abs(actual) > 1e-3
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((pred[mask] - actual[mask]) / np.abs(actual[mask])) * 100)


# ── Build detailed results table ───────────────────────────────────────────────

def build_results(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        pred = predict_pipeline(
            payload,
            model      = row["model_short"],
            imgsz      = int(row["imgsz"]),
            precision  = row["precision"],
            target_fps = float(row["target_fps"]),
            width      = int(row["width"]),
            height     = int(row["height"]),
        )

        # Actual T_frame = sum of measured stage latencies
        actual_T_frame = (
            row["capture_lat_ms"] + row["preprocess_lat_ms"] +
            row["infer_lat_ms"]   + row["postprocess_lat_ms"]
        )

        r = {
            # Identity
            "model":      row["model_short"],
            "imgsz":      int(row["imgsz"]),
            "precision":  row["precision"],
            "target_fps": row["target_fps"],
            "config":     config_key(pd.DataFrame([row])).iloc[0],

            # ── Latency: predicted vs actual (ms) ──────────────────────────
            "pred_T_capture_ms":    pred["T_capture_ms"],
            "actual_T_capture_ms":  row["capture_lat_ms"],
            "pred_T_preprocess_ms": pred["T_preprocess_ms"],
            "actual_T_preprocess_ms": row["preprocess_lat_ms"],
            "pred_T_infer_ms":      pred["T_infer_ms"],
            "actual_T_infer_ms":    row["infer_lat_ms"],
            "pred_T_postprocess_ms":pred["T_postprocess_ms"],
            "actual_T_postprocess_ms": row["postprocess_lat_ms"],
            "pred_T_frame_ms":      pred["T_frame_ms"],
            "actual_T_frame_ms":    actual_T_frame,

            # ── Energy: predicted vs actual (mJ/frame) ─────────────────────
            "pred_E_capture_mj":    pred["E_capture_mj"],
            "actual_E_capture_mj":  row["capture_e_mj"],
            "pred_E_preprocess_mj": pred["E_preprocess_mj"],
            "actual_E_preprocess_mj": row["preprocess_e_mj"],
            "pred_E_infer_mj":      pred["E_infer_mj"],
            "actual_E_infer_mj":    row["infer_energy_mj"],
            "pred_E_postprocess_mj":pred["E_postprocess_mj"],
            "actual_E_postprocess_mj": row["postprocess_energy_mj"],
            "pred_E_overhead_mj":   pred["E_overhead_mj"],
            "actual_E_idle_mj":     row["idle_mj"],
            "pred_E_total_mj":      pred["E_total_mj"],
            "actual_E_total_mj":    row["energy_per_frame_mj"],

            # ── End-to-end ─────────────────────────────────────────────────
            "pred_fps":    pred["fps"],
            "actual_fps":  row["fps_mean"],
            "bottleneck":  pred["bottleneck"],

            # ── Errors ─────────────────────────────────────────────────────
            "fps_err_pct": (pred["fps"] - row["fps_mean"])
                           / max(row["fps_mean"], 1e-6) * 100,
            "E_total_err_pct": (pred["E_total_mj"] - row["energy_per_frame_mj"])
                               / max(row["energy_per_frame_mj"], 1e-6) * 100,
        }

        # Per-stage latency and energy errors
        for stage, (t_pred_key, e_pred_key), (t_act_key, e_act_key) in [
            ("capture",    ("T_capture_ms",    "E_capture_mj"),
                           ("capture_lat_ms",  "capture_e_mj")),
            ("preprocess", ("T_preprocess_ms", "E_preprocess_mj"),
                           ("preprocess_lat_ms","preprocess_e_mj")),
            ("infer",      ("T_infer_ms",       "E_infer_mj"),
                           ("infer_lat_ms",     "infer_energy_mj")),
            ("postprocess",("T_postprocess_ms", "E_postprocess_mj"),
                           ("postprocess_lat_ms","postprocess_energy_mj")),
        ]:
            t_pred = pred[t_pred_key]
            t_act  = row[t_act_key]
            e_pred = pred[e_pred_key]
            e_act  = row[e_act_key]

            r[f"{stage}_T_err_pct"] = (
                (t_pred - t_act) / max(abs(t_act), 1e-3) * 100
                if abs(t_act) > 1e-3 else float("nan")
            )
            r[f"{stage}_E_err_pct"] = (
                (e_pred - e_act) / max(abs(e_act), 1e-3) * 100
                if abs(e_act) > 1e-3 else float("nan")
            )

        rows.append(r)

    return pd.DataFrame(rows)


# ── Summaries ──────────────────────────────────────────────────────────────────

def summary_by_config(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, grp in results.groupby("config"):
        row = {"config": cfg, "n": len(grp)}
        for col in ["fps_err_pct", "E_total_err_pct",
                    "capture_T_err_pct", "preprocess_T_err_pct",
                    "infer_T_err_pct",   "postprocess_T_err_pct",
                    "capture_E_err_pct", "preprocess_E_err_pct",
                    "infer_E_err_pct",   "postprocess_E_err_pct"]:
            vals = grp[col].dropna()
            row[f"{col}_MAPE"] = np.mean(np.abs(vals)) if len(vals) else float("nan")
            row[f"{col}_bias"] = np.mean(vals)          if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_by_fps(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fps, grp in results.groupby("target_fps"):
        row = {"target_fps": fps, "n": len(grp)}
        for col in ["fps_err_pct", "E_total_err_pct",
                    "infer_T_err_pct", "capture_T_err_pct"]:
            vals = grp[col].dropna()
            row[f"{col}_MAPE"] = np.mean(np.abs(vals)) if len(vals) else float("nan")
            row[f"{col}_bias"] = np.mean(vals)          if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def stage_error_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        for metric, label in [("T", "Latency (ms)"), ("E", "Energy (mJ)")]:
            col = f"{stage}_{metric}_err_pct"
            vals = results[col].dropna()
            rows.append({
                "stage":   stage,
                "metric":  label,
                "MAPE_%":  round(np.mean(np.abs(vals)), 1) if len(vals) else float("nan"),
                "bias_%":  round(np.mean(vals), 1)          if len(vals) else float("nan"),
                "std_%":   round(np.std(vals), 1)            if len(vals) else float("nan"),
                "n_valid": int(len(vals)),
            })
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_stage_breakdown(results: pd.DataFrame, out_dir: Path) -> None:
    """
    For each config at target_fps=0, stacked bars side by side:
    predicted stage breakdown vs actual stage breakdown.
    """
    sub = results[results["target_fps"] == 0].copy()
    configs = sorted(sub["config"].unique())
    n = len(configs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage Breakdown — Predicted vs Actual (unbounded FPS)", fontsize=12)

    for ax, metric, stages_cols in [
        (axes[0], "Energy / frame (mJ)", [
            ("capture",    "E_capture_mj"),
            ("preprocess", "E_preprocess_mj"),
            ("infer",      "E_infer_mj"),
            ("postprocess","E_postprocess_mj"),
            ("overhead",   "E_overhead_mj"),
        ]),
        (axes[1], "Frame latency (ms)", [
            ("capture",    "T_capture_ms"),
            ("preprocess", "T_preprocess_ms"),
            ("infer",      "T_infer_ms"),
            ("postprocess","T_postprocess_ms"),
        ]),
    ]:
        x = np.arange(n)
        width = 0.35
        # Predicted (left bar) and Actual (right bar)
        for bar_offset, prefix, label_suffix in [(-width/2, "pred", "Pred"), (width/2, "actual", "Actual")]:
            bottom = np.zeros(n)
            for stage, col_base in stages_cols:
                col = f"{prefix}_{col_base}"
                if col not in sub.columns:
                    continue
                agg = sub.groupby("config")[col].median().reindex(configs).fillna(0).values
                ax.bar(x + bar_offset, agg, width, bottom=bottom,
                       color=STAGE_COLORS.get(stage, "#999"),
                       alpha=0.9 if label_suffix == "Pred" else 0.5,
                       label=f"{stage} ({label_suffix})" if bar_offset == -width/2 else None)
                bottom += agg

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        # Legend: one entry per stage, no duplicates
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [l.split(" (")[0] for l in labels], fontsize=8)
        ax.set_title(f"{metric}\n(left=predicted, right=actual)")

    fig.tight_layout()
    p = out_dir / "stage_breakdown_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_error_by_fps(results: pd.DataFrame, out_dir: Path) -> None:
    """
    Line plots: MAPE for FPS and energy as a function of target_fps,
    split by config.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Prediction Error vs Target FPS", fontsize=12)

    colors = plt.cm.tab10.colors
    configs = sorted(results["config"].unique())

    for ax, err_col, ylabel in [
        (axes[0], "fps_err_pct",    "FPS error (%)"),
        (axes[1], "E_total_err_pct","Energy/frame error (%)"),
    ]:
        fps_vals = sorted(results["target_fps"].unique())
        for i, cfg in enumerate(configs):
            sub = results[results["config"] == cfg]
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
    p = out_dir / "error_by_fps.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_stage_error_heatmap(results: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap: stage × config, cell = mean |error %| for energy.
    Highlights which stage is most fragile for each config.
    """
    configs = sorted(results["config"].unique())
    stage_energy_cols = [
        ("capture",    "capture_E_err_pct"),
        ("preprocess", "preprocess_E_err_pct"),
        ("infer",      "infer_E_err_pct"),
        ("postprocess","postprocess_E_err_pct"),
        ("end-to-end", "E_total_err_pct"),
    ]

    mat = np.full((len(stage_energy_cols), len(configs)), float("nan"))
    for j, cfg in enumerate(configs):
        sub = results[results["config"] == cfg]
        for i, (_, col) in enumerate(stage_energy_cols):
            vals = sub[col].dropna()
            if len(vals) > 0:
                mat[i, j] = np.mean(np.abs(vals))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.8), 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=30)
    plt.colorbar(im, ax=ax, label="Mean |error| (%)")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(stage_energy_cols)))
    ax.set_yticklabels([s for s, _ in stage_energy_cols])
    ax.set_title("Stage Energy Prediction Error — Mean |error %| per Config\n"
                 "(green = accurate, red = fragile)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if v > 20 else "black")

    fig.tight_layout()
    p = out_dir / "stage_error_heatmap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_pred_vs_actual_scatter(results: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: predicted vs actual for FPS and total energy (all rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predicted vs Actual — All Rows (in-sample)", fontsize=12)

    colors = plt.cm.tab10.colors
    configs = sorted(results["config"].unique())
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(configs)}

    for ax, pred_col, actual_col, label, unit in [
        (axes[0], "pred_fps",       "actual_fps",       "FPS",          "fps"),
        (axes[1], "pred_E_total_mj","actual_E_total_mj","Energy/frame", "mJ"),
    ]:
        for cfg in configs:
            sub = results[results["config"] == cfg]
            ax.scatter(sub[actual_col], sub[pred_col],
                       label=cfg, color=color_map[cfg], alpha=0.75, s=30)

        lo = min(results[actual_col].min(), results[pred_col].min()) * 0.9
        hi = max(results[actual_col].max(), results[pred_col].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")

        mape = safe_mape(results[actual_col].values, results[pred_col].values)
        mae  = np.mean(np.abs(results[pred_col].values - results[actual_col].values))
        ax.set_title(f"{label}\nMAPE={mape:.1f}%  MAE={mae:.2f} {unit}")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p = out_dir / "pred_vs_actual_scatter.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}\n")

    payload = joblib.load(PREDICTOR_PATH)
    df = load_data()

    print("Building per-row predictions...")
    results = build_results(df, payload)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    results.to_csv(OUT_DIR / "detailed_results.csv", index=False)
    print(f"  Saved detailed_results.csv  ({len(results)} rows)")

    cfg_summary = summary_by_config(results)
    cfg_summary.to_csv(OUT_DIR / "summary_by_config.csv", index=False)

    fps_summary = summary_by_fps(results)
    fps_summary.to_csv(OUT_DIR / "summary_by_fps.csv", index=False)

    stage_summary = stage_error_summary(results)
    stage_summary.to_csv(OUT_DIR / "stage_errors.csv", index=False)

    # ── Print summary tables ───────────────────────────────────────────────────
    print("\n── Overall stage errors (in-sample) ────────────────────────")
    print(stage_summary.to_string(index=False))

    print("\n── End-to-end error per config ─────────────────────────────")
    cols = ["config", "n", "fps_err_pct_MAPE", "fps_err_pct_bias",
            "E_total_err_pct_MAPE", "E_total_err_pct_bias"]
    print(cfg_summary[cols].round(1).to_string(index=False))

    print("\n── End-to-end error per target FPS ─────────────────────────")
    cols = ["target_fps", "n", "fps_err_pct_MAPE", "E_total_err_pct_MAPE",
            "fps_err_pct_bias", "E_total_err_pct_bias"]
    print(fps_summary[cols].round(1).to_string(index=False))

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────────")
    plot_stage_breakdown(results, OUT_DIR)
    plot_error_by_fps(results, OUT_DIR)
    plot_stage_error_heatmap(results, OUT_DIR)
    plot_pred_vs_actual_scatter(results, OUT_DIR)

    print(f"\nDone. All outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
