import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "training_data"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_PATH = MODEL_DIR / "energy_cpu_linear.joblib"


def load_training_data() -> pd.DataFrame:
    """
    Load and concatenate all CSVs from data/training_data.

    This script assumes you manually curate which CSVs should be used for
    training by copying/moving them into data/training_data/.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data directory not found: {DATA_DIR}")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print("Using training CSVs:")
    for csv_path in csv_files:
        print(f"  - {csv_path}")

    frames = [pd.read_csv(p) for p in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    include_model_feature: bool,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a simple feature matrix X and target y for energy prediction.

    Predict CPU energy per run (`energy_cpu_J`) from static
    model / hyperparameter-derived features (no measured latency):
    model identity, FLOPs, batch size, input resolution, and precision.
    """
    required_columns = [target_col, "flops_total", "batch", "resolution", "precision"]
    if include_model_feature:
        required_columns.append("model")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    # Basic coercion / cleaning.
    df = df.copy()
    for col in ("energy_cpu_J", "flops_total", "batch", "resolution"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["precision"] = df["precision"].astype(str).str.lower().str.strip()
    df = df[df["precision"] != ""]
    if include_model_feature:
        df["model"] = df["model"].astype(str).str.lower().str.strip()
        df = df[df["model"] != ""]
    df = df.dropna(subset=required_columns)

    numeric_feature_cols = ["flops_total", "batch", "resolution"]
    X_num = df[numeric_feature_cols]
    X_precision = pd.get_dummies(df["precision"], prefix="precision")
    feature_frames = [X_num, X_precision]
    if include_model_feature:
        X_model = pd.get_dummies(df["model"], prefix="model")
        feature_frames.append(X_model)
    X = pd.concat(feature_frames, axis=1)
    y = df[target_col]
    return X, y


def _create_estimator(estimator: str):
    if estimator == "linear":
        return LinearRegression()
    if estimator == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
    raise ValueError("Unsupported estimator. Use one of: linear, hgb.")


def _infer_model_task_from_name(model_name: str) -> str:
    name = model_name.lower()
    if "ssd" in name or "yolo" in name or "rcnn" in name:
        return "detection"
    return "classification"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CPU energy predictor from curated training CSVs."
    )
    parser.add_argument(
        "--include-model-feature",
        action="store_true",
        help="Include one-hot encoded model identity columns in X.",
    )
    parser.add_argument(
        "--separate-by",
        type=str,
        default="model_task",
        choices=["model_task", "model"],
        help="Train one regressor per model_task (default) or per model.",
    )
    args = parser.parse_args()

    estimator = "hgb"
    use_log_target = True
    target_cols = ["energy_cpu_J", "energy_gpu_J", "energy_io_J", "energy_total_J"]
    try:
        df = load_training_data()
        if "energy_total_J" not in df.columns:
            component_cols = ["energy_cpu_J", "energy_gpu_J", "energy_io_J"]
            if all(col in df.columns for col in component_cols):
                df = df.copy()
                for col in component_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["energy_total_J"] = df[component_cols].sum(axis=1, min_count=1)

        if "model_task" not in df.columns:
            if "model" not in df.columns:
                raise ValueError(
                    "Training data must include at least one of: model_task, model."
                )
            df = df.copy()
            df["model_task"] = df["model"].astype(str).map(_infer_model_task_from_name)

        if args.separate_by == "model" and "model" not in df.columns:
            raise ValueError("Training data must include 'model' to use --separate-by model.")

        group_col = args.separate_by
        missing_targets = [c for c in target_cols if c not in df.columns]
        if missing_targets:
            raise ValueError(
                "Training data is missing required energy targets: "
                f"{missing_targets}. Expected: {target_cols}"
            )

        models_by_group_by_target: dict[str, dict[str, object]] = {
            target: {} for target in target_cols
        }
        feature_names_by_group_by_target: dict[str, dict[str, list[str]]] = {
            target: {} for target in target_cols
        }
        metrics_rows: list[dict[str, object]] = []

        for target_col in target_cols:
            for group_name_raw, group_df in df.groupby(group_col):
                group_name = str(group_name_raw).strip().lower()
                if not group_name:
                    continue

                group_df = group_df.copy()
                X, y = build_feature_matrix(
                    group_df,
                    include_model_feature=(
                        args.include_model_feature and args.separate_by != "model"
                    ),
                    target_col=target_col,
                )
                if len(X) < 6:
                    print(
                        f"Skipping target='{target_col}' {group_col}='{group_name}' "
                        f"due to too few rows for train/test split: {len(X)}"
                    )
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
                model = _create_estimator(estimator)

                y_train_fit = np.log1p(y_train) if use_log_target else y_train
                model.fit(X_train, y_train_fit)
                y_pred_raw = model.predict(X_test)
                y_pred = np.expm1(y_pred_raw) if use_log_target else y_pred_raw
                y_pred = np.maximum(y_pred, 0.0)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                metrics_rows.append(
                    {
                        "target": target_col,
                        "group_by": group_col,
                        "group": group_name,
                        "rows": len(X),
                        "r2": r2,
                        "mae_j": mae,
                    }
                )
                models_by_group_by_target[target_col][group_name] = model
                feature_names_by_group_by_target[target_col][group_name] = list(X.columns)

        trained_targets = [
            target for target, grouped in models_by_group_by_target.items() if grouped
        ]
        if not trained_targets:
            raise ValueError("No group-specific models were trained for any target.")

        print(f"Per-{group_col}, per-target metrics:")
        for m in metrics_rows:
            print(
                f"  target={m['target']} {group_col}={m['group']} rows={m['rows']} "
                f"R^2={m['r2']:.4f} MAE={m['mae_j']:.6f} J"
            )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cpu_models = models_by_group_by_target.get("energy_cpu_J", {})
        cpu_features = feature_names_by_group_by_target.get("energy_cpu_J", {})
        payload = {
            "models_by_group_by_target": models_by_group_by_target,
            "feature_names_by_group_by_target": feature_names_by_group_by_target,
            "group_by": group_col,
            "targets": trained_targets,
            "target_transforms": {
                target: ("log1p" if use_log_target else "none") for target in trained_targets
            },
            # Backward-compatible aliases for older predictor versions.
            "models_by_group": cpu_models if cpu_models else None,
            "feature_names_by_group": cpu_features if cpu_features else None,
            "models_by_task": cpu_models if group_col == "model_task" else None,
            "feature_names_by_task": cpu_features if group_col == "model_task" else None,
            "target": "energy_cpu_J",
            "target_transform": "log1p" if use_log_target else "none",
            "estimator": estimator,
            "include_model_feature": bool(
                args.include_model_feature and args.separate_by != "model"
            ),
            "groups": sorted(
                {
                    group
                    for grouped_models in models_by_group_by_target.values()
                    for group in grouped_models.keys()
                }
            ),
        }
        dump(payload, MODEL_PATH)
        print(f"Saved model to: {MODEL_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error while training energy model: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

