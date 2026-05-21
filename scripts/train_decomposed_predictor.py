"""
Train a decomposed energy predictor for the PyTorch camera pipeline.

Instead of one black-box model that takes FPS as an input, this trains
one small sub-predictor per pipeline stage:

  Stage 1 — Capture     : cap.read() incl. camera wait
  Stage 2 — Preprocess  : resize + normalise + CPU→GPU memcpy
  Stage 3 — Inference   : model forward pass (CUDA)
  Stage 4 — Postprocess : NMS / bounding-box decode (CPU)
  Stage 5 — Overhead    : idle background draw (constant power)

Each sub-predictor outputs:
  - T_stage_ms  : latency per frame (ms)
  - E_stage_mj  : energy per frame (mJ)

The combination layer then derives:
  T_frame = T_capture + T_preprocess + T_infer + T_postprocess
  FPS     = 1000 / T_frame
  E_total = E_capture + E_preprocess + E_infer + E_postprocess
          + P_overhead * T_frame          (overhead scales with frame time)

FPS is therefore a *prediction output*, not an input.

Data source
-----------
  results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_133405/sweep_summary.csv
  results/camera_bench/yolov8n_fps_sweep_MAXN_20260428_150803/sweep_summary.csv
  results/camera_bench/ssdlite_fps_sweep_MAXN_20260428_162918/sweep_summary.csv

Each CSV already contains per-stage latency and energy columns produced by
camera_bench.metrics.attribute_stage_energy() during the sweep.

Output
------
  models/decomposed_predictor.pkl          — all sub-predictors + metadata
  results/analysis/decomposed_predictor_<ts>/
    cv_results.csv
    stage_breakdown.png                    — measured stage contributions
    stage_pred_vs_actual.png               — per-stage predicted vs actual
    end_to_end_validation.png              — combined FPS and energy vs measured
    overhead_estimate.png                  — P_overhead constant
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_DIRS = [
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_133405",
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_150803",
    PROJECT_ROOT / "results" / "camera_bench" / "ssdlite_fps_sweep_MAXN_20260428_162918",
]

MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "decomposed_predictor.pkl"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / f"decomposed_predictor_{TIMESTAMP}"

# Long model name → short label
MODEL_LABELS: dict[str, str] = {
    "yolov8n": "yolov8n",
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
    "ssdlite320": "ssdlite320",
}

# Model family used for postprocess predictor
MODEL_FAMILY: dict[str, str] = {
    "yolov8n": "yolo",         # fused pre+infer+post → postprocess = 0
    "ssdlite320": "torchvision",  # separate postprocess stage
}

STAGES = ["capture", "preprocess", "infer", "postprocess"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and merge all sweep CSVs. Returns only ok rows with stage data."""
    frames: list[pd.DataFrame] = []
    for d in SWEEP_DIRS:
        csv = d / "sweep_summary.csv"
        if not csv.exists():
            print(f"WARNING: {csv} not found — skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        if "yolo_imgsz" not in df.columns:
            df["yolo_imgsz"] = 640
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No sweep CSVs found — check SWEEP_DIRS.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].copy()

    # Normalise model names
    df["model_short"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["model_family"] = df["model_short"].map(MODEL_FAMILY).fillna("unknown")
    df["is_fp16"] = (df["precision"].str.lower() == "fp16").astype(float)
    df["imgsz"] = pd.to_numeric(df["yolo_imgsz"], errors="coerce").fillna(640).astype(float)

    # ── Unify inference columns (YOLO fused vs torchvision separate) ──────────
    # YOLO: infer_fused_lat covers preprocess+infer+postprocess together.
    #       We assign it to infer and set postprocess=0.
    # Torchvision: infer_lat covers the forward pass only; postprocess is separate.

    is_yolo = df["model_family"] == "yolo"

    df["infer_lat_ms"] = np.where(
        is_yolo,
        pd.to_numeric(df.get("infer_fused_lat_mean_ms", np.nan), errors="coerce"),
        pd.to_numeric(df.get("infer_lat_mean_ms", np.nan), errors="coerce"),
    )
    df["infer_energy_j"] = np.where(
        is_yolo,
        pd.to_numeric(df.get("infer_fused_energy_j", np.nan), errors="coerce"),
        pd.to_numeric(df.get("infer_energy_j", np.nan), errors="coerce"),
    )
    df["postprocess_lat_ms"] = np.where(
        is_yolo,
        0.0,  # already included in infer_fused
        pd.to_numeric(df.get("postprocess_lat_mean_ms", np.nan), errors="coerce"),
    )
    df["postprocess_energy_j"] = np.where(
        is_yolo,
        0.0,
        pd.to_numeric(df.get("postprocess_energy_j", np.nan), errors="coerce"),
    )

    # Rename existing columns for consistency
    df["capture_lat_ms"]    = pd.to_numeric(df.get("capture_lat_mean_ms",    np.nan), errors="coerce")
    df["preprocess_lat_ms"] = pd.to_numeric(df.get("preprocess_lat_mean_ms", np.nan), errors="coerce")
    df["capture_e_j"]       = pd.to_numeric(df.get("capture_energy_j",       np.nan), errors="coerce")
    df["preprocess_e_j"]    = pd.to_numeric(df.get("preprocess_energy_j",    np.nan), errors="coerce")
    df["idle_j"]            = pd.to_numeric(df.get("idle_j",                 np.nan), errors="coerce")

    # Drop rows where any stage data is missing
    stage_cols = [
        "capture_lat_ms", "preprocess_lat_ms", "infer_lat_ms", "postprocess_lat_ms",
        "capture_e_j",    "preprocess_e_j",    "infer_energy_j", "postprocess_energy_j",
        "fps_mean", "energy_per_frame_j",
    ]
    n_before = len(df)
    df = df.dropna(subset=stage_cols).copy()
    print(f"Loaded {n_before} ok rows → {len(df)} rows with complete stage data.")

    # Convert energy to mJ for readability
    for col in ["capture_e_j", "preprocess_e_j", "infer_energy_j", "postprocess_energy_j",
                "idle_j", "energy_per_frame_j"]:
        df[col.replace("_j", "_mj")] = df[col] * 1000

    return df


def config_key(df: pd.DataFrame) -> pd.Series:
    return df["model_short"] + "_imgsz" + df["imgsz"].astype(int).astype(str) + "_" + df["precision"].str.lower()


# ── Stage feature builders ─────────────────────────────────────────────────────

def capture_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture stage: depends on target_fps and camera resolution.
    At target_fps=0 (unbounded), cap.read() blocks until the camera delivers a frame.
    At target_fps < camera_fps, the pipeline throttles via a sleep → cap.read() returns
    immediately so T_capture ≈ 0, but the sleep time is embedded elsewhere.

    We model capture_lat directly as a function of target_fps.
    """
    feat = pd.DataFrame(index=df.index)
    feat["width"]      = df["width"].astype(float)
    feat["height"]     = df["height"].astype(float)
    feat["pixels"]     = feat["width"] * feat["height"]
    feat["target_fps"] = pd.to_numeric(df["target_fps"], errors="coerce").fillna(0)
    # At target_fps=0, encode as "run as fast as possible" = 30
    feat["eff_fps"]    = feat["target_fps"].replace(0, 30).astype(float)
    return feat


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess: resize (width×height → imgsz×imgsz) + normalise + CPU→GPU memcpy.
    Scales with input pixel count and output size.
    """
    feat = pd.DataFrame(index=df.index)
    feat["in_pixels"]  = df["width"].astype(float) * df["height"].astype(float)
    feat["out_pixels"] = df["imgsz"].astype(float) ** 2
    feat["is_fp16"]    = df["is_fp16"]
    return feat


def infer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference: model forward pass on CUDA.
    Depends on model (one-hot), imgsz, and precision.
    """
    feat = pd.DataFrame(index=df.index)
    for label in ["yolov8n", "ssdlite320"]:
        feat[f"model_{label}"] = (df["model_short"] == label).astype(float)
    feat["imgsz"]   = df["imgsz"]
    feat["is_fp16"] = df["is_fp16"]
    return feat


def postprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Postprocess: NMS + box decode.
    For YOLO this is fused into inference (always 0 here).
    For torchvision models it is a separate CPU step.
    """
    feat = pd.DataFrame(index=df.index)
    feat["is_torchvision"] = (df["model_family"] == "torchvision").astype(float)
    return feat


STAGE_CFG = {
    #  name            features_fn         lat_col              energy_col
    "capture":    (capture_features,    "capture_lat_ms",    "capture_e_mj"),
    "preprocess": (preprocess_features, "preprocess_lat_ms", "preprocess_e_mj"),
    "infer":      (infer_features,      "infer_lat_ms",      "infer_energy_mj"),
    "postprocess":(postprocess_features,"postprocess_lat_ms","postprocess_energy_mj"),
}


# ── Model builder ──────────────────────────────────────────────────────────────

def make_model() -> Pipeline:
    """Degree-2 polynomial ridge — good for small datasets with nonlinear interactions."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge",  Ridge(alpha=1.0)),
    ])


# ── Leave-one-config-out CV ────────────────────────────────────────────────────

def loocv_stage(
    df: pd.DataFrame,
    features_fn,
    lat_col: str,
    energy_col: str,
    stage_name: str,
) -> pd.DataFrame:
    """
    Run leave-one-config-out CV for one stage.
    Returns a DataFrame with per-fold metrics for both lat and energy targets.
    """
    configs = config_key(df)
    unique_configs = sorted(configs.unique())
    X = features_fn(df)

    rows = []
    for hold in unique_configs:
        test_mask  = configs == hold
        train_mask = ~test_mask
        X_tr, X_te = X[train_mask], X[test_mask]

        for target_col, unit, scale in [
            (lat_col,    "ms", 1.0),
            (energy_col, "mJ", 1.0),
        ]:
            y_tr = df.loc[train_mask, target_col]
            y_te = df.loc[test_mask,  target_col]

            if y_tr.isna().any() or y_te.isna().any():
                continue
            if y_tr.std() < 1e-9:
                # Constant target (e.g. postprocess for YOLO = 0)
                y_pred = np.zeros(len(y_te))
            else:
                m = make_model()
                m.fit(X_tr, y_tr)
                y_pred = np.maximum(m.predict(X_te), 0.0)

            mae  = mean_absolute_error(y_te, y_pred)
            mape = mean_absolute_percentage_error(y_te + 1e-9, y_pred + 1e-9) * 100
            r2   = r2_score(y_te, y_pred) if len(y_te) > 1 else float("nan")

            rows.append({
                "stage":    stage_name,
                "target":   target_col,
                "held_out": hold,
                "n_test":   int(test_mask.sum()),
                "mae":      mae,
                "mape_pct": mape,
                "r2":       r2,
            })
            print(
                f"  [{stage_name:12s}] hold={hold:<40s}  "
                f"target={target_col:<25s}  "
                f"MAE={mae*scale:7.2f}{unit}  MAPE={mape:5.1f}%"
            )

    return pd.DataFrame(rows)


# ── Final model training ───────────────────────────────────────────────────────

def train_stage(
    df: pd.DataFrame,
    features_fn,
    lat_col: str,
    energy_col: str,
) -> dict:
    """Train final models on all data for one stage. Returns {target: model}."""
    X = features_fn(df)
    models = {}
    for target_col in [lat_col, energy_col]:
        y = df[target_col]
        if y.isna().any():
            models[target_col] = None
            continue
        if y.std() < 1e-9:
            # Constant (e.g., postprocess latency for YOLO = 0)
            models[target_col] = "zero"
            continue
        m = make_model()
        m.fit(X, y)
        models[target_col] = m
    return models


# ── Overhead estimation ────────────────────────────────────────────────────────

def estimate_overhead(df: pd.DataFrame) -> float:
    """
    Estimate the background idle power draw (W).

    idle_j is the energy attributed to the idle period within each timed window
    by attribute_stage_energy(). P_overhead = idle_j / T_frame.
    """
    T_frame_s = 1.0 / df["fps_mean"].clip(lower=0.1)
    P_idle = df["idle_mj"] / 1000.0 / T_frame_s  # W
    P_idle = P_idle[np.isfinite(P_idle) & (P_idle > 0)]
    p_overhead = float(P_idle.median())
    print(f"\nOverhead power estimate:  median={p_overhead:.3f} W  "
          f"(std={P_idle.std():.3f} W,  n={len(P_idle)})")
    return p_overhead


# ── Combination layer (also saved in payload for inference) ───────────────────

def predict_pipeline(
    payload: dict,
    model: str,
    imgsz: int,
    precision: str,
    target_fps: float = 0,
    width: int = 640,
    height: int = 480,
) -> dict:
    """
    Predict FPS and per-stage energy for a given configuration.

    Parameters
    ----------
    payload     : loaded from models/decomposed_predictor.pkl
    model       : 'yolov8n' or 'ssdlite320'
    imgsz       : model input size in pixels (320 or 640)
    precision   : 'fp32' or 'fp16'
    target_fps  : 0 = unbounded, >0 = FPS cap
    width/height: camera capture resolution

    Returns
    -------
    dict with keys:
        fps, T_frame_ms, E_total_mj,
        E_capture_mj, T_capture_ms,
        E_preprocess_mj, T_preprocess_ms,
        E_infer_mj, T_infer_ms,
        E_postprocess_mj, T_postprocess_ms,
        E_overhead_mj,
        bottleneck ('throttle' or 'compute')
    """
    model_labels = payload["model_labels"]
    model_family = payload["model_family"]
    stage_models = payload["stage_models"]
    p_overhead   = payload["p_overhead_w"]

    model_short = model_labels.get(model, model)
    family      = model_family.get(model_short, "unknown")
    is_fp16     = float(precision.lower() == "fp16")
    eff_fps     = float(target_fps) if target_fps > 0 else 30.0

    # Build one-row DataFrames for each stage's feature function
    row = pd.DataFrame([{
        "model_short": model_short,
        "model_family": family,
        "is_fp16": is_fp16,
        "imgsz": float(imgsz),
        "width": float(width),
        "height": float(height),
        "target_fps": float(target_fps),
        "eff_fps": eff_fps,
    }])

    results: dict[str, float] = {}
    feature_fns = payload["feature_fns"]

    for stage, (lat_col, energy_col) in payload["stage_col_map"].items():
        feat_fn  = feature_fns[stage]
        X_pred   = feat_fn(row)
        s_models = stage_models[stage]

        for col_key, out_key in [(lat_col, f"T_{stage}_ms"), (energy_col, f"E_{stage}_mj")]:
            m = s_models.get(col_key)
            if m is None or m == "zero":
                results[out_key] = 0.0
            else:
                results[out_key] = float(np.maximum(m.predict(X_pred)[0], 0.0))

    # Compute combined outputs
    T_compute = (results["T_capture_ms"] + results["T_preprocess_ms"] +
                 results["T_infer_ms"]   + results["T_postprocess_ms"])

    if target_fps > 0:
        T_frame = max(T_compute, 1000.0 / target_fps)
    else:
        T_frame = T_compute

    fps = 1000.0 / max(T_frame, 1e-3)

    E_overhead = p_overhead * (T_frame / 1000.0) * 1000  # mJ
    E_total    = (results["E_capture_mj"] + results["E_preprocess_mj"] +
                  results["E_infer_mj"]   + results["E_postprocess_mj"] + E_overhead)

    return {
        "fps":              round(fps, 2),
        "T_frame_ms":       round(T_frame, 2),
        "E_total_mj":       round(E_total, 1),
        "E_capture_mj":     round(results["E_capture_mj"], 1),
        "T_capture_ms":     round(results["T_capture_ms"], 2),
        "E_preprocess_mj":  round(results["E_preprocess_mj"], 1),
        "T_preprocess_ms":  round(results["T_preprocess_ms"], 2),
        "E_infer_mj":       round(results["E_infer_mj"], 1),
        "T_infer_ms":       round(results["T_infer_ms"], 2),
        "E_postprocess_mj": round(results["E_postprocess_mj"], 1),
        "T_postprocess_ms": round(results["T_postprocess_ms"], 2),
        "E_overhead_mj":    round(E_overhead, 1),
        "bottleneck":       "throttle" if T_frame > T_compute else "compute",
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "capture":     "#4C72B0",
    "preprocess":  "#DD8452",
    "infer":       "#55A868",
    "postprocess": "#C44E52",
    "overhead":    "#8172B2",
}

CONFIG_COLORS = {
    "yolov8n_imgsz640_fp32":    "#1f77b4",
    "yolov8n_imgsz640_fp16":    "#aec7e8",
    "yolov8n_imgsz320_fp32":    "#ff7f0e",
    "yolov8n_imgsz320_fp16":    "#ffbb78",
    "ssdlite320_imgsz640_fp32": "#2ca02c",
}


def plot_stage_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar: stage contribution to latency and energy per config at unbounded FPS."""
    sub = df[df["target_fps"] == 0].copy()
    configs = config_key(sub)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, unit, cols in [
        (axes[0], "Latency",       "ms",
         [("capture_lat_ms","Capture"), ("preprocess_lat_ms","Preprocess"),
          ("infer_lat_ms","Inference"), ("postprocess_lat_ms","Postprocess")]),
        (axes[1], "Energy/frame",  "mJ",
         [("capture_e_mj","Capture"), ("preprocess_e_mj","Preprocess"),
          ("infer_energy_mj","Inference"), ("postprocess_energy_mj","Postprocess"),
          ("idle_mj","Overhead")]),
    ]:
        # Aggregate: median per config
        agg = sub.groupby(configs)[
            [c for c, _ in cols]
        ].median()

        x = np.arange(len(agg))
        bottom = np.zeros(len(agg))
        for col, label in cols:
            stage = col.split("_")[0]
            vals = agg[col].values
            ax.bar(x, vals, bottom=bottom,
                   label=label,
                   color=STAGE_COLORS.get(stage, "#999999"),
                   alpha=0.85)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"{metric} ({unit})")
        ax.set_title(f"Stage {metric} Breakdown — unbounded FPS")
        ax.legend(fontsize=8)

    fig.tight_layout()
    p = out_dir / "stage_breakdown.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_stage_pred_vs_actual(
    df: pd.DataFrame,
    stage_models: dict,
    out_dir: Path,
) -> None:
    """Per-stage: predicted vs actual for both latency and energy."""
    feature_fns = {
        "capture":    capture_features,
        "preprocess": preprocess_features,
        "infer":      infer_features,
        "postprocess":postprocess_features,
    }
    stage_col_map = {
        "capture":    ("capture_lat_ms",    "capture_e_mj"),
        "preprocess": ("preprocess_lat_ms", "preprocess_e_mj"),
        "infer":      ("infer_lat_ms",      "infer_energy_mj"),
        "postprocess":("postprocess_lat_ms","postprocess_energy_mj"),
    }

    configs = config_key(df)
    n_stages = len(STAGES)
    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 8))
    fig.suptitle("Stage Predictors — Predicted vs Actual", fontsize=12)

    for col_i, stage in enumerate(STAGES):
        feat_fn  = feature_fns[stage]
        lat_col, energy_col = stage_col_map[stage]
        X = feat_fn(df)
        s_models = stage_models[stage]

        for row_i, (target_col, unit) in enumerate([
            (lat_col, "ms"), (energy_col, "mJ")
        ]):
            ax = axes[row_i][col_i]
            m = s_models.get(target_col)

            if m is None or m == "zero":
                ax.set_title(f"{stage}\n{target_col}\n(constant=0)")
                ax.axis("off")
                continue

            y_true = df[target_col].values
            y_pred = np.maximum(m.predict(X), 0.0)

            for cfg in sorted(configs.unique()):
                mask = (configs == cfg).values
                ax.scatter(
                    y_true[mask], y_pred[mask],
                    label=cfg, color=CONFIG_COLORS.get(cfg, "#999999"),
                    alpha=0.7, s=25,
                )

            lim_lo = min(y_true.min(), y_pred.min()) * 0.9
            lim_hi = max(y_true.max(), y_pred.max()) * 1.05
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)

            mape = mean_absolute_percentage_error(y_true + 1e-9, y_pred + 1e-9) * 100
            ax.set_title(f"{stage}\n{target_col}\nMAPE={mape:.1f}%", fontsize=8)
            ax.set_xlabel(f"Actual ({unit})", fontsize=8)
            ax.set_ylabel(f"Predicted ({unit})", fontsize=8)
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    p = out_dir / "stage_pred_vs_actual.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_end_to_end_validation(
    df: pd.DataFrame,
    payload: dict,
    out_dir: Path,
) -> None:
    """
    Compare combined pipeline predictions against measured values.
    One point per (config, target_fps) cell (median over repeats).
    """
    configs = config_key(df)

    # Build predictions for every row
    pred_fps, pred_e = [], []
    for _, row in df.iterrows():
        p = predict_pipeline(
            payload,
            model=row["model_short"],
            imgsz=int(row["imgsz"]),
            precision=row["precision"],
            target_fps=float(row["target_fps"]),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        pred_fps.append(p["fps"])
        pred_e.append(p["E_total_mj"])

    df = df.copy()
    df["pred_fps"]   = pred_fps
    df["pred_e_mj"]  = pred_e
    df["actual_e_mj"] = df["energy_per_frame_j"] * 1000

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("End-to-End Validation — Combined Pipeline Prediction", fontsize=12)

    for ax, (actual_col, pred_col, label, unit) in zip(axes, [
        ("fps_mean",    "pred_fps",  "FPS",          "fps"),
        ("actual_e_mj", "pred_e_mj", "Energy/frame", "mJ"),
    ]):
        y_true = df[actual_col].values
        y_pred = df[pred_col].values

        for cfg in sorted(configs.unique()):
            mask = (configs == cfg).values
            ax.scatter(
                y_true[mask], y_pred[mask],
                label=cfg, color=CONFIG_COLORS.get(cfg, "#999999"),
                alpha=0.75, s=30,
            )

        lim_lo = min(y_true.min(), y_pred.min()) * 0.9
        lim_hi = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="perfect")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)

        mae  = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true + 1e-9, y_pred + 1e-9) * 100
        ax.set_xlabel(f"Actual ({unit})", fontsize=10)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=10)
        ax.set_title(f"{label}\nMAE={mae:.2f} {unit}  MAPE={mape:.1f}%")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p = out_dir / "end_to_end_validation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_cv_summary(cv_all: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: mean MAPE per stage across held-out configs."""
    summary = (
        cv_all.groupby(["stage", "target"])["mape_pct"]
        .mean()
        .reset_index()
        .sort_values("mape_pct")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = summary["stage"] + "\n" + summary["target"].str.split("_").str[0]
    bars = ax.bar(range(len(summary)), summary["mape_pct"],
                  color="#4C72B0", alpha=0.8)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title("Leave-one-config-out CV — Mean MAPE per Stage Target")
    for bar, v in zip(bars, summary["mape_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = out_dir / "cv_mape_summary.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    # 1. Load data
    df = load_data()
    print(f"\nConfigs in dataset:")
    for cfg, grp in df.groupby(config_key(df)):
        fps_vals = grp["fps_mean"].values
        e_vals   = grp["energy_per_frame_j"].values * 1000
        print(f"  {cfg:<45s}  n={len(grp):2d}  "
              f"fps={fps_vals.mean():.1f}  "
              f"E/frame={e_vals.mean():.0f}mJ")

    # 2. Estimate overhead
    p_overhead = estimate_overhead(df)

    stage_col_map = {
        "capture":    ("capture_lat_ms",    "capture_e_mj"),
        "preprocess": ("preprocess_lat_ms", "preprocess_e_mj"),
        "infer":      ("infer_lat_ms",      "infer_energy_mj"),
        "postprocess":("postprocess_lat_ms","postprocess_energy_mj"),
    }
    feature_fns = {
        "capture":    capture_features,
        "preprocess": preprocess_features,
        "infer":      infer_features,
        "postprocess":postprocess_features,
    }

    # 3. Leave-one-config-out CV per stage
    print("\n── Leave-one-config-out CV ─────────────────────────────────")
    cv_frames = []
    for stage in STAGES:
        print(f"\n  Stage: {stage}")
        feat_fn, lat_col, energy_col = (
            feature_fns[stage],
            stage_col_map[stage][0],
            stage_col_map[stage][1],
        )
        cv_df = loocv_stage(df, feat_fn, lat_col, energy_col, stage)
        cv_frames.append(cv_df)

    cv_all = pd.concat(cv_frames, ignore_index=True)
    cv_csv = OUT_DIR / "cv_results.csv"
    cv_all.to_csv(cv_csv, index=False)
    print(f"\nSaved CV results → {cv_csv}")

    print("\n── CV summary (mean MAPE per stage) ────────────────────────")
    summary = cv_all.groupby(["stage", "target"])["mape_pct"].mean().round(1)
    print(summary.to_string())

    # 4. Train final models on all data
    print("\n── Training final models on all data ───────────────────────")
    stage_models: dict[str, dict] = {}
    for stage in STAGES:
        feat_fn, lat_col, energy_col = (
            feature_fns[stage],
            stage_col_map[stage][0],
            stage_col_map[stage][1],
        )
        stage_models[stage] = train_stage(df, feat_fn, lat_col, energy_col)
        print(f"  {stage}: trained")

    # 5. Save payload
    payload = {
        "stage_models":   stage_models,
        "stage_col_map":  stage_col_map,
        "feature_fns":    feature_fns,
        "p_overhead_w":   p_overhead,
        "model_labels":   MODEL_LABELS,
        "model_family":   MODEL_FAMILY,
        "cv_summary":     summary,
        "timestamp":      TIMESTAMP,
    }
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_SAVE_PATH)
    print(f"\nSaved predictor → {MODEL_SAVE_PATH}")

    # 6. Print example predictions
    print("\n── Example predictions ─────────────────────────────────────")
    for model, imgsz, prec, fps in [
        ("yolov8n",    640, "fp32", 0),
        ("yolov8n",    640, "fp16", 0),
        ("ssdlite320", 640, "fp32", 0),
        ("yolov8n",    640, "fp32", 10),
        ("ssdlite320", 640, "fp32", 10),
    ]:
        p = predict_pipeline(payload, model, imgsz, prec, fps)
        print(
            f"  {model:<12s} imgsz={imgsz} {prec:<4s} "
            f"target_fps={fps:2d}  →  "
            f"predicted FPS={p['fps']:5.1f}  "
            f"E/frame={p['E_total_mj']:6.0f}mJ  "
            f"bottleneck={p['bottleneck']}"
        )

    # 7. Plots
    print("\n── Generating plots ────────────────────────────────────────")
    plot_stage_breakdown(df, OUT_DIR)
    plot_stage_pred_vs_actual(df, stage_models, OUT_DIR)
    plot_end_to_end_validation(df, payload, OUT_DIR)
    plot_cv_summary(cv_all, OUT_DIR)

    print(f"\nDone. All outputs in:\n  {OUT_DIR}")
    print(f"Predictor saved to:\n  {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
