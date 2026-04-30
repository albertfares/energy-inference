"""
Train an energy predictor for the camera benchmark pipeline.

Data source: the three sweep_summary.csv files produced under
  results/camera_bench/{yolov8n_fps_sweep_640,yolov8n_fps_sweep_320,ssdlite_fps_sweep}/

Features (available *before* running a benchmark):
  model      — one-hot:  yolov8n | ssdlite320
  imgsz      — numeric:  320 | 640
  is_fp16    — binary:   0 (fp32) | 1 (fp16)
  actual_fps — numeric:  measured fps_mean (reflects saturation ceiling naturally)

Targets:
  energy_per_frame_j  (primary)
  mean_power_w        (secondary)

Models evaluated:
  poly2   — Pipeline(PolynomialFeatures(degree=2, interaction_only=False), Ridge)
  rf      — RandomForestRegressor

Evaluation:
  Leave-one-config-out cross-validation (5 configs × 21 runs each = 105 rows).
  A "config" is (model, imgsz, precision).

Output:
  models/camera_energy_predictor.pkl    — best model (lowest MAPE on CV)
  results/analysis/camera_predictor_<timestamp>/
    cv_results.csv
    predicted_vs_actual.png
    residuals.png
    feature_importance.png   (RF only)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_DIRS = [
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_133405",
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_150803",
    PROJECT_ROOT / "results" / "camera_bench" / "ssdlite_fps_sweep_MAXN_20260428_162918",
]

MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "camera_energy_predictor.pkl"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / f"camera_predictor_{TIMESTAMP}"

# model name → short label used as feature
MODEL_LABELS = {
    "yolov8n": "yolov8n",
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
}

TARGETS = ["energy_per_frame_j", "mean_power_w"]
PRIMARY_TARGET = "energy_per_frame_j"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sweep_data() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in SWEEP_DIRS:
        csv = d / "sweep_summary.csv"
        if not csv.exists():
            print(f"WARNING: {csv} not found, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        # First YOLO sweep (imgsz=640) was recorded before yolo_imgsz column was added.
        if "yolo_imgsz" not in df.columns:
            df["yolo_imgsz"] = 640
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No sweep CSVs found — check SWEEP_DIRS paths.")

    combined = pd.concat(frames, ignore_index=True)
    # Keep only successful runs with energy data.
    combined = combined[combined["status"] == "ok"].copy()
    combined = combined.dropna(subset=TARGETS + ["fps_mean"])
    print(f"Loaded {len(combined)} rows from {len(frames)} CSVs.")
    return combined


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame aligned to df's index."""
    feat = pd.DataFrame(index=df.index)

    # Model one-hot
    short_name = df["model"].map(MODEL_LABELS).fillna(df["model"])
    for label in ["yolov8n", "ssdlite320"]:
        feat[f"model_{label}"] = (short_name == label).astype(float)

    # Resolution
    feat["imgsz"] = pd.to_numeric(df["yolo_imgsz"], errors="coerce").fillna(640).astype(float)

    # Precision
    feat["is_fp16"] = (df["precision"].str.lower() == "fp16").astype(float)

    # Actual FPS (captures saturation ceiling)
    feat["actual_fps"] = pd.to_numeric(df["fps_mean"], errors="coerce")

    return feat


def config_key(df: pd.DataFrame) -> pd.Series:
    """Return a string key per row identifying its (model, imgsz, precision) config."""
    return (
        df["model"].map(MODEL_LABELS).fillna(df["model"])
        + "_imgsz" + df["yolo_imgsz"].astype(str)
        + "_" + df["precision"].str.lower()
    )


# ── Model builders ─────────────────────────────────────────────────────────────

def make_poly2() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=10.0)),
    ])


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


ESTIMATORS = {
    "poly2": make_poly2,
    "rf": make_rf,
}


# ── Leave-one-config-out CV ────────────────────────────────────────────────────

def run_loocv(
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    For each (estimator_name, target) pair, run leave-one-config-out CV.

    Returns a dict: {estimator_name: DataFrame of per-fold metrics}.
    """
    configs = config_key(df)
    unique_configs = sorted(configs.unique())
    print(f"\nLeave-one-config-out CV: {len(unique_configs)} folds")
    for c in unique_configs:
        print(f"  {c}: {(configs == c).sum()} rows")

    all_results: dict[str, list[dict]] = {name: [] for name in ESTIMATORS}

    for hold_config in unique_configs:
        test_mask = configs == hold_config
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]

        for est_name, est_factory in ESTIMATORS.items():
            for target in TARGETS:
                y_train = df.loc[train_mask, target]
                y_test = df.loc[test_mask, target]

                est = est_factory()
                est.fit(X_train, y_train)
                y_pred = est.predict(X_test)
                y_pred = np.maximum(y_pred, 0.0)

                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                r2 = r2_score(y_test, y_pred)

                all_results[est_name].append({
                    "held_out": hold_config,
                    "estimator": est_name,
                    "target": target,
                    "n_test": int(test_mask.sum()),
                    "mae": mae,
                    "mape_pct": mape,
                    "r2": r2,
                })

                unit = "mJ" if target == "energy_per_frame_j" else "W"
                scale = 1000 if target == "energy_per_frame_j" else 1
                print(
                    f"  [{est_name:6s}] hold={hold_config:<40s} "
                    f"target={target:<22s} "
                    f"MAE={mae*scale:6.1f}{unit:3s}  "
                    f"MAPE={mape:5.1f}%  R²={r2:.3f}"
                )

    return {
        name: pd.DataFrame(rows)
        for name, rows in all_results.items()
    }


# ── Final model training (all data) ───────────────────────────────────────────

def train_final_models(
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Train each estimator on all data for all targets. Return {est_name: {target: model}}."""
    final: dict[str, dict[str, object]] = {}
    for est_name, est_factory in ESTIMATORS.items():
        final[est_name] = {}
        for target in TARGETS:
            y = df[target]
            est = est_factory()
            est.fit(X, y)
            final[est_name][target] = est
    return final


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    "yolov8n_imgsz640_fp32":   "#1f77b4",
    "yolov8n_imgsz640_fp16":   "#aec7e8",
    "yolov8n_imgsz320_fp32":   "#ff7f0e",
    "yolov8n_imgsz320_fp16":   "#ffbb78",
    "ssdlite320_imgsz640_fp32": "#2ca02c",
}

DEFAULT_COLOR = "#999999"


def plot_predicted_vs_actual(
    X: pd.DataFrame,
    df: pd.DataFrame,
    final_models: dict[str, dict[str, object]],
    out_dir: Path,
) -> None:
    configs = config_key(df)

    for est_name, target_models in final_models.items():
        for target in TARGETS:
            model = target_models[target]
            y_true = df[target].values
            y_pred = np.maximum(model.predict(X), 0.0)

            scale = 1000 if target == "energy_per_frame_j" else 1
            unit = "mJ/frame" if target == "energy_per_frame_j" else "W"

            fig, ax = plt.subplots(figsize=(6, 6))
            for cfg in sorted(configs.unique()):
                mask = (configs == cfg).values
                color = COLORS.get(cfg, DEFAULT_COLOR)
                ax.scatter(
                    y_true[mask] * scale,
                    y_pred[mask] * scale,
                    label=cfg,
                    color=color,
                    alpha=0.75,
                    s=30,
                )

            lims = [
                min(y_true.min(), y_pred.min()) * scale * 0.9,
                max(y_true.max(), y_pred.max()) * scale * 1.05,
            ]
            ax.plot(lims, lims, "k--", lw=1, label="perfect")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f"Actual ({unit})")
            ax.set_ylabel(f"Predicted ({unit})")
            ax.set_title(f"{est_name} — {target}\n(trained on all data)")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = out_dir / f"pred_vs_actual_{est_name}_{target}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {fname.name}")


def plot_residuals(
    X: pd.DataFrame,
    df: pd.DataFrame,
    final_models: dict[str, dict[str, object]],
    out_dir: Path,
) -> None:
    configs = config_key(df)

    for est_name, target_models in final_models.items():
        for target in TARGETS:
            model = target_models[target]
            y_true = df[target].values
            y_pred = np.maximum(model.predict(X), 0.0)
            residuals_pct = (y_pred - y_true) / y_true * 100

            fig, ax = plt.subplots(figsize=(7, 4))
            for cfg in sorted(configs.unique()):
                mask = (configs == cfg).values
                color = COLORS.get(cfg, DEFAULT_COLOR)
                actual_fps = X.loc[configs == cfg, "actual_fps"].values
                ax.scatter(
                    actual_fps,
                    residuals_pct[mask],
                    label=cfg,
                    color=color,
                    alpha=0.75,
                    s=30,
                )

            ax.axhline(0, color="k", linestyle="--", lw=1)
            ax.set_xlabel("Actual FPS")
            ax.set_ylabel("Residual (%)")
            ax.set_title(f"{est_name} — {target} residuals vs FPS\n(trained on all data)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = out_dir / f"residuals_{est_name}_{target}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {fname.name}")


def plot_cv_summary(cv_results: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Bar chart: MAPE per held-out config, per estimator, for primary target."""
    fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(5 * len(ESTIMATORS), 5), sharey=True)
    if len(ESTIMATORS) == 1:
        axes = [axes]

    for ax, (est_name, df_cv) in zip(axes, cv_results.items()):
        sub = df_cv[df_cv["target"] == PRIMARY_TARGET].sort_values("held_out")
        bars = ax.bar(range(len(sub)), sub["mape_pct"], color="#4C72B0", alpha=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["held_out"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{est_name}\nmean MAPE = {sub['mape_pct'].mean():.1f}%")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, sub["mape_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"Leave-one-config-out CV — {PRIMARY_TARGET}", fontsize=11)
    fig.tight_layout()
    fname = out_dir / "cv_mape_summary.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname.name}")


def plot_feature_importance(
    X: pd.DataFrame,
    final_models: dict[str, dict[str, object]],
    out_dir: Path,
) -> None:
    rf_models = final_models.get("rf")
    if not rf_models:
        return
    model = rf_models[PRIMARY_TARGET]
    importances = model.feature_importances_
    feature_names = X.columns.tolist()
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[order], color="#4C72B0", alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Feature importance (mean decrease impurity)")
    ax.set_title(f"RF feature importance — {PRIMARY_TARGET}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "rf_feature_importance.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname.name}")


# ── Select best estimator ──────────────────────────────────────────────────────

def select_best_estimator(cv_results: dict[str, pd.DataFrame]) -> str:
    """Return the estimator name with the lowest mean MAPE on the primary target."""
    best_name, best_mape = None, float("inf")
    for name, df_cv in cv_results.items():
        mean_mape = df_cv[df_cv["target"] == PRIMARY_TARGET]["mape_pct"].mean()
        print(f"  {name:8s}  mean MAPE on {PRIMARY_TARGET} = {mean_mape:.2f}%")
        if mean_mape < best_mape:
            best_mape = mean_mape
            best_name = name
    print(f"→ Best estimator: {best_name}  (mean MAPE = {best_mape:.2f}%)")
    return best_name


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # 1. Load data
    df = load_sweep_data()

    # 2. Build features
    X = build_features(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print("Features:", X.columns.tolist())
    print("\nData summary:")
    configs = config_key(df)
    for cfg in sorted(configs.unique()):
        n = (configs == cfg).sum()
        y = df.loc[configs == cfg, PRIMARY_TARGET]
        print(f"  {cfg:<45s}  n={n:2d}  {PRIMARY_TARGET}: "
              f"min={y.min()*1000:.0f}mJ  max={y.max()*1000:.0f}mJ  mean={y.mean()*1000:.0f}mJ")

    # 3. Leave-one-config-out CV
    print("\n── CV ──────────────────────────────────────────────────────")
    cv_results = run_loocv(X, df)

    # 4. Save CV results
    all_cv = pd.concat(cv_results.values(), ignore_index=True)
    cv_csv = OUT_DIR / "cv_results.csv"
    all_cv.to_csv(cv_csv, index=False)
    print(f"\nSaved CV results → {cv_csv}")

    # 5. Summary table
    print("\n── CV summary (mean over held-out configs) ─────────────────")
    summary = (
        all_cv.groupby(["estimator", "target"])[["mae", "mape_pct", "r2"]]
        .mean()
        .round({"mae": 4, "mape_pct": 2, "r2": 3})
    )
    print(summary.to_string())

    # 6. Select best
    print("\n── Best estimator ──────────────────────────────────────────")
    best_name = select_best_estimator(cv_results)

    # 7. Train final models on all data
    print("\n── Training final models on all data ───────────────────────")
    final_models = train_final_models(X, df)

    # 8. Save predictor payload
    feature_names = X.columns.tolist()
    payload = {
        "best_estimator": best_name,
        "models": {
            est_name: {
                target: model
                for target, model in target_models.items()
            }
            for est_name, target_models in final_models.items()
        },
        "feature_names": feature_names,
        "targets": TARGETS,
        "model_labels": MODEL_LABELS,
        "cv_summary": summary,
        "cv_timestamp": TIMESTAMP,
    }
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_SAVE_PATH)
    print(f"Saved predictor → {MODEL_SAVE_PATH}")

    # 9. Plots
    print("\n── Generating plots ────────────────────────────────────────")
    plot_predicted_vs_actual(X, df, final_models, OUT_DIR)
    plot_residuals(X, df, final_models, OUT_DIR)
    plot_cv_summary(cv_results, OUT_DIR)
    plot_feature_importance(X, final_models, OUT_DIR)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
