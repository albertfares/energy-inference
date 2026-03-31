import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "training_data" / "filtred_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "energy_cpu_linear.joblib"
DEFAULT_TARGET = "energy_cpu_J"
SUPPORTED_TARGETS = ["energy_cpu_J", "energy_gpu_J", "energy_io_J", "energy_total_J"]
TARGET_CHOICES = SUPPORTED_TARGETS + ["all"]


def infer_model_task(model_name: str) -> str:
    name = model_name.lower()
    if "ssd" in name or "yolo" in name or "rcnn" in name:
        return "detection"
    return "classification"


def _prepare_group_rows(
    group_df: pd.DataFrame,
    *,
    include_model_feature: bool,
    target_col: str,
) -> pd.DataFrame:
    required_columns = [target_col, "flops_total", "batch", "resolution", "precision"]
    if include_model_feature:
        required_columns.append("model")

    missing = [c for c in required_columns if c not in group_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for split reconstruction: {missing}")

    out = group_df.copy()
    for col in (target_col, "flops_total", "batch", "resolution"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["precision"] = out["precision"].astype(str).str.lower().str.strip()
    out = out[out["precision"] != ""]

    if include_model_feature:
        out["model"] = out["model"].astype(str).str.lower().str.strip()
        out = out[out["model"] != ""]

    out = out.dropna(subset=required_columns)
    return out


def reconstruct_split_rows(
    df: pd.DataFrame,
    *,
    separate_by: str,
    include_model_feature: bool,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if separate_by not in {"model", "model_task"}:
        raise ValueError("separate_by must be one of: model, model_task")

    work = df.copy()
    if "model_task" not in work.columns:
        if "model" not in work.columns:
            raise ValueError("Input data must include at least one of: model_task, model.")
        work["model_task"] = work["model"].astype(str).map(infer_model_task)

    work = work.reset_index(drop=False).rename(columns={"index": "source_row_idx"})
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []

    for group_name_raw, group_df in work.groupby(separate_by):
        group_name = str(group_name_raw).strip().lower()
        if not group_name:
            continue

        clean_group = _prepare_group_rows(
            group_df,
            include_model_feature=(include_model_feature and separate_by != "model"),
            target_col=target_col,
        )

        if len(clean_group) < 6:
            skipped_rows.append(
                {
                    "group_by": separate_by,
                    "group": group_name,
                    "rows_after_cleaning": len(clean_group),
                    "reason": "too_few_rows_for_split",
                }
            )
            continue

        idx = np.arange(len(clean_group))
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)
        train_part = clean_group.iloc[train_idx].copy()
        test_part = clean_group.iloc[test_idx].copy()
        train_part["split_group"] = group_name
        test_part["split_group"] = group_name
        train_frames.append(train_part)
        test_frames.append(test_part)

    if not train_frames or not test_frames:
        raise ValueError("No groups produced a valid train/test split.")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    skipped_df = pd.DataFrame(skipped_rows)
    return train_df, test_df, skipped_df


def _select_model_and_features(payload: dict, row: pd.Series, target_name: str):
    group_by = payload.get("group_by")
    models_by_group_by_target = payload.get("models_by_group_by_target")
    feature_names_by_group_by_target = payload.get("feature_names_by_group_by_target")
    models_by_group = payload.get("models_by_group")
    feature_names_by_group = payload.get("feature_names_by_group")
    models_by_task = payload.get("models_by_task")
    feature_names_by_task = payload.get("feature_names_by_task")
    model_single = payload.get("model")
    feature_names_single = payload.get("feature_names")

    model_name = str(row["model"]).strip().lower()
    task_name = str(row["model_task"]).strip().lower()

    # Multi-target payload.
    if (
        isinstance(models_by_group_by_target, dict)
        and isinstance(feature_names_by_group_by_target, dict)
    ):
        grouped_models = models_by_group_by_target.get(target_name)
        grouped_features = feature_names_by_group_by_target.get(target_name)
        if not isinstance(grouped_models, dict) or not isinstance(grouped_features, dict):
            return None, None
        key = model_name if group_by == "model" else task_name
        if key not in grouped_models or key not in grouped_features:
            return None, None
        return grouped_models[key], grouped_features[key]

    # Backward-compatible single-target payload.
    payload_target = str(payload.get("target", DEFAULT_TARGET))
    if target_name != payload_target:
        return None, None

    if (
        group_by is not None
        and isinstance(models_by_group, dict)
        and isinstance(feature_names_by_group, dict)
    ):
        key = model_name if group_by == "model" else task_name
        if key not in models_by_group or key not in feature_names_by_group:
            return None, None
        return models_by_group[key], feature_names_by_group[key]

    if isinstance(models_by_task, dict) and isinstance(feature_names_by_task, dict):
        if task_name not in models_by_task or task_name not in feature_names_by_task:
            return None, None
        return models_by_task[task_name], feature_names_by_task[task_name]

    if model_single is None or feature_names_single is None:
        return None, None
    return model_single, feature_names_single


def _predict_row(model, feature_names: list[str], row: pd.Series, target_transform: str) -> float:
    payload_row = {
        "flops_total": float(row["flops_total"]),
        "batch": int(row["batch"]),
        "resolution": int(row["resolution"]),
    }
    model_key = f"model_{str(row['model']).strip().lower()}"
    precision_key = f"precision_{str(row['precision']).strip().lower()}"
    for feature in feature_names:
        if feature.startswith("model_"):
            payload_row[feature] = 1.0 if feature == model_key else 0.0
        elif feature.startswith("precision_"):
            payload_row[feature] = 1.0 if feature == precision_key else 0.0

    x = pd.DataFrame([payload_row]).reindex(columns=feature_names, fill_value=0.0)
    y_pred = float(model.predict(x)[0])
    if target_transform == "log1p":
        y_pred = math.expm1(y_pred)
    return max(y_pred, 0.0)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    if len(df) == 0:
        return {
            "count": 0.0,
            "mae_j": float("nan"),
            "rmse_j": float("nan"),
            "r2": float("nan"),
            "mape_pct": float("nan"),
        }
    return {
        "count": float(len(df)),
        "mae_j": float(mean_absolute_error(y_true, y_pred)),
        "rmse_j": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(df) >= 2 else float("nan"),
        "mape_pct": _safe_mape(y_true, y_pred),
    }


def _aggregate_with_baseline(
    df: pd.DataFrame,
    *,
    level_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_df in df.groupby("model"):
        baseline = _compute_metrics(model_df)
        baseline_mae = baseline["mae_j"]
        for level_value, sub_df in model_df.groupby(level_col):
            metrics = _compute_metrics(sub_df)
            delta_pct = float("nan")
            if (
                isinstance(baseline_mae, float)
                and not np.isnan(baseline_mae)
                and baseline_mae > 0
                and isinstance(metrics["mae_j"], float)
                and not np.isnan(metrics["mae_j"])
            ):
                delta_pct = (metrics["mae_j"] / baseline_mae - 1.0) * 100.0

            rows.append(
                {
                    "model": model_name,
                    level_col: level_value,
                    "count": int(metrics["count"]),
                    "mae_j": metrics["mae_j"],
                    "rmse_j": metrics["rmse_j"],
                    "r2": metrics["r2"],
                    "mape_pct": metrics["mape_pct"],
                    "mae_delta_pct_vs_model": delta_pct,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", level_col]).reset_index(drop=True)


def _aggregate_full_config(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["model", "batch", "resolution", "precision"]
    for model_name, model_df in df.groupby("model"):
        baseline = _compute_metrics(model_df)
        baseline_mae = baseline["mae_j"]
        for (model, batch, resolution, precision), sub_df in model_df.groupby(group_cols):
            metrics = _compute_metrics(sub_df)
            mean_signed_err = float(sub_df["signed_err_j"].mean()) if len(sub_df) > 0 else float("nan")
            delta_pct = float("nan")
            if (
                isinstance(baseline_mae, float)
                and not np.isnan(baseline_mae)
                and baseline_mae > 0
                and isinstance(metrics["mae_j"], float)
                and not np.isnan(metrics["mae_j"])
            ):
                delta_pct = (metrics["mae_j"] / baseline_mae - 1.0) * 100.0

            rows.append(
                {
                    "model": model,
                    "batch": int(batch),
                    "resolution": int(resolution),
                    "precision": str(precision),
                    "count": int(metrics["count"]),
                    "mae_j": metrics["mae_j"],
                    "rmse_j": metrics["rmse_j"],
                    "r2": metrics["r2"],
                    "mape_pct": metrics["mape_pct"],
                    "mean_signed_err_j": mean_signed_err,
                    "mae_delta_pct_vs_model": delta_pct,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", "batch", "resolution", "precision"]).reset_index(drop=True)


def _select_best_worst_configs(
    full_cfg_df: pd.DataFrame,
    *,
    min_count: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if full_cfg_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    filtered = full_cfg_df[full_cfg_df["count"] >= min_count].copy()
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    best_frames: list[pd.DataFrame] = []
    worst_frames: list[pd.DataFrame] = []
    for model_name, model_df in filtered.groupby("model"):
        best = model_df.sort_values(["mae_j", "rmse_j", "mape_pct"]).head(top_n).copy()
        best["rank_within_model"] = np.arange(1, len(best) + 1)
        best_frames.append(best)

        worst = model_df.sort_values(["mae_j", "rmse_j", "mape_pct"], ascending=False).head(top_n).copy()
        worst["rank_within_model"] = np.arange(1, len(worst) + 1)
        worst_frames.append(worst)

    best_df = pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()
    worst_df = pd.concat(worst_frames, ignore_index=True) if worst_frames else pd.DataFrame()
    return best_df, worst_df


def _aggregate_pairwise(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_df in df.groupby("model"):
        baseline = _compute_metrics(model_df)
        baseline_mae = baseline["mae_j"]
        for (a, b), sub_df in model_df.groupby([col_a, col_b]):
            metrics = _compute_metrics(sub_df)
            delta_pct = float("nan")
            if (
                isinstance(baseline_mae, float)
                and not np.isnan(baseline_mae)
                and baseline_mae > 0
                and isinstance(metrics["mae_j"], float)
                and not np.isnan(metrics["mae_j"])
            ):
                delta_pct = (metrics["mae_j"] / baseline_mae - 1.0) * 100.0

            rows.append(
                {
                    "model": model_name,
                    col_a: a,
                    col_b: b,
                    "count": int(metrics["count"]),
                    "mae_j": metrics["mae_j"],
                    "rmse_j": metrics["rmse_j"],
                    "r2": metrics["r2"],
                    "mape_pct": metrics["mape_pct"],
                    "mae_delta_pct_vs_model": delta_pct,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", col_a, col_b]).reset_index(drop=True)


def _weighted_mean_variance(values: pd.Series, weights: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    w = weights.to_numpy(dtype=float)
    x = values.to_numpy(dtype=float)
    if np.sum(w) <= 0:
        return 0.0
    mu = float(np.sum(w * x) / np.sum(w))
    return float(np.sum(w * (x - mu) ** 2) / np.sum(w))


def _interaction_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate relative importance of single factors and pairwise interactions
    for absolute error, using weighted variance of subgroup mean errors.
    """
    rows: list[dict[str, object]] = []
    effect_defs = [
        ("batch", ["batch"]),
        ("resolution", ["resolution"]),
        ("precision", ["precision"]),
        ("batch_x_resolution", ["batch", "resolution"]),
        ("batch_x_precision", ["batch", "precision"]),
        ("resolution_x_precision", ["resolution", "precision"]),
        ("full_config", ["batch", "resolution", "precision"]),
    ]

    for model_name, model_df in df.groupby("model"):
        abs_err = model_df["abs_err_j"].to_numpy(dtype=float)
        total_var = float(np.var(abs_err)) if len(abs_err) > 0 else 0.0
        score_map: dict[str, float] = {}
        for effect_name, cols in effect_defs:
            grouped = (
                model_df.groupby(cols)["abs_err_j"]
                .agg(["mean", "count"])
                .reset_index(drop=False)
            )
            score = _weighted_mean_variance(grouped["mean"], grouped["count"])
            score_map[effect_name] = max(score, 0.0)

        score_sum = float(sum(score_map.values()))
        for effect_name in score_map:
            share = (score_map[effect_name] / score_sum * 100.0) if score_sum > 0 else 0.0
            rows.append(
                {
                    "model": model_name,
                    "effect": effect_name,
                    "score": score_map[effect_name],
                    "share_pct": share,
                    "abs_err_variance_total": total_var,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", "share_pct"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct train/test split used by training script and evaluate predictor "
            "on reconstructed holdout rows."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help="Training CSV used to fit the predictor (default: data/training_data/filtred_data.csv).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to saved predictor payload (.joblib).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/analysis/predictor_holdout_<timestamp>/",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        choices=TARGET_CHOICES,
        help="Target energy column to evaluate, or 'all' (default: energy_cpu_J).",
    )
    parser.add_argument(
        "--min-count-config",
        type=int,
        default=2,
        help="Minimum samples per joint config to include in best/worst rankings (default: 2).",
    )
    parser.add_argument(
        "--top-n-configs",
        type=int,
        default=5,
        help="Top-N best/worst joint configurations per model (default: 5).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    model_path = Path(args.model_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model payload not found: {model_path}")

    df_raw = pd.read_csv(input_path)
    payload = load(model_path)
    if not isinstance(payload, dict):
        raise ValueError("Loaded model payload is not a dict.")

    separate_by = str(payload.get("group_by") or "model_task")
    include_model_feature = bool(payload.get("include_model_feature", False))
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = PROJECT_ROOT / "results" / "analysis" / f"predictor_holdout_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    available_targets = payload.get("targets")
    if not isinstance(available_targets, list) or not available_targets:
        available_targets = [str(payload.get("target", DEFAULT_TARGET))]
    available_targets = [str(t) for t in available_targets if str(t) in SUPPORTED_TARGETS]
    if not available_targets:
        raise ValueError("No supported targets found in model payload.")

    requested_target = str(args.target)
    if requested_target == "all":
        targets_to_run = available_targets
    else:
        targets_to_run = [requested_target]
        if requested_target not in available_targets:
            raise ValueError(
                f"Requested target '{requested_target}' not available in payload. "
                f"Available: {available_targets}"
            )

    target_transforms = payload.get("target_transforms")
    default_transform = str(payload.get("target_transform", "none"))

    for target_col in targets_to_run:
        df_target = df_raw.copy()
        if target_col == "energy_total_J" and target_col not in df_target.columns:
            component_cols = ["energy_cpu_J", "energy_gpu_J", "energy_io_J"]
            if all(col in df_target.columns for col in component_cols):
                for col in component_cols:
                    df_target[col] = pd.to_numeric(df_target[col], errors="coerce")
                df_target[target_col] = df_target[component_cols].sum(axis=1, min_count=1)
        if target_col not in df_target.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in input CSV after preprocessing."
            )

        if isinstance(target_transforms, dict):
            target_transform = str(target_transforms.get(target_col, default_transform))
        else:
            payload_target = str(payload.get("target", DEFAULT_TARGET))
            if payload_target != target_col:
                raise ValueError(
                    f"Loaded payload appears single-target ('{payload_target}') but "
                    f"requested target is '{target_col}'. Retrain with multi-target support "
                    "or run with --target matching the payload target."
                )
            target_transform = default_transform

        target_out_dir = out_root / target_col if requested_target == "all" else out_root
        target_out_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df, skipped_df = reconstruct_split_rows(
            df_target,
            separate_by=separate_by,
            include_model_feature=include_model_feature,
            target_col=target_col,
        )

        train_df.to_csv(target_out_dir / "reconstructed_train_rows.csv", index=False)
        test_df.to_csv(target_out_dir / "reconstructed_test_rows.csv", index=False)
        if not skipped_df.empty:
            skipped_df.to_csv(target_out_dir / "skipped_groups.csv", index=False)

        pred_rows: list[dict[str, object]] = []
        missing_group = 0
        test_rows = test_df.to_dict("records")
        for row_data in tqdm(
            test_rows,
            desc=f"Scoring holdout rows [{target_col}]",
            unit="row",
        ):
            row = pd.Series(row_data)
            selected_model, selected_feature_names = _select_model_and_features(
                payload,
                row,
                target_name=target_col,
            )
            if selected_model is None or selected_feature_names is None:
                missing_group += 1
                continue

            y_true = float(row[target_col])
            y_pred = _predict_row(
                selected_model,
                list(selected_feature_names),
                row,
                target_transform=target_transform,
            )
            abs_err = abs(y_pred - y_true)
            pred_rows.append(
                {
                    "model": str(row["model"]).strip().lower(),
                    "model_task": str(row["model_task"]).strip().lower(),
                    "batch": int(row["batch"]),
                    "resolution": int(row["resolution"]),
                    "precision": str(row["precision"]).strip().lower(),
                    "source_row_idx": int(row["source_row_idx"]),
                    "target": target_col,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "signed_err_j": y_pred - y_true,
                    "abs_err_j": abs_err,
                    "pct_err": (abs_err / y_true * 100.0) if y_true != 0 else float("nan"),
                }
            )

        pred_df = pd.DataFrame(pred_rows)
        if pred_df.empty:
            raise ValueError("No predictions were produced. Check group alignment in payload/data.")
        pred_df.to_csv(target_out_dir / "holdout_predictions.csv", index=False)

        overall_metrics = pd.DataFrame([_compute_metrics(pred_df)])
        overall_metrics.to_csv(target_out_dir / "metrics_overall.csv", index=False)

        by_model_rows = []
        for model_name, g in pred_df.groupby("model"):
            m = _compute_metrics(g)
            m["model"] = model_name
            by_model_rows.append(m)
        metrics_by_model = pd.DataFrame(by_model_rows).sort_values("model")
        metrics_by_model.to_csv(target_out_dir / "metrics_by_model.csv", index=False)

        _aggregate_with_baseline(pred_df, level_col="batch").to_csv(
            target_out_dir / "predictor_error_by_model_batch.csv", index=False
        )
        _aggregate_with_baseline(pred_df, level_col="resolution").to_csv(
            target_out_dir / "predictor_error_by_model_resolution.csv", index=False
        )
        _aggregate_with_baseline(pred_df, level_col="precision").to_csv(
            target_out_dir / "predictor_error_by_model_precision.csv", index=False
        )
        full_cfg_df = _aggregate_full_config(pred_df)
        full_cfg_df.to_csv(target_out_dir / "predictor_error_by_model_full_config.csv", index=False)

        best_cfg_df, worst_cfg_df = _select_best_worst_configs(
            full_cfg_df,
            min_count=max(args.min_count_config, 1),
            top_n=max(args.top_n_configs, 1),
        )
        best_cfg_df.to_csv(target_out_dir / "predictor_best_configs_per_model.csv", index=False)
        worst_cfg_df.to_csv(target_out_dir / "predictor_worst_configs_per_model.csv", index=False)

        _aggregate_pairwise(pred_df, "batch", "resolution").to_csv(
            target_out_dir / "predictor_error_by_model_batch_resolution.csv", index=False
        )
        _aggregate_pairwise(pred_df, "batch", "precision").to_csv(
            target_out_dir / "predictor_error_by_model_batch_precision.csv", index=False
        )
        _aggregate_pairwise(pred_df, "resolution", "precision").to_csv(
            target_out_dir / "predictor_error_by_model_resolution_precision.csv", index=False
        )
        _interaction_importance(pred_df).to_csv(
            target_out_dir / "predictor_interaction_importance_per_model.csv", index=False
        )

        print(f"Input CSV: {input_path}")
        print(f"Model payload: {model_path}")
        print(f"target={target_col} (transform={target_transform})")
        print(f"group_by={separate_by}, include_model_feature={include_model_feature}")
        print(f"Reconstructed train rows: {len(train_df)}")
        print(f"Reconstructed test rows: {len(test_df)}")
        print(f"Predicted holdout rows: {len(pred_df)}")
        if missing_group > 0:
            print(f"Skipped rows due to missing group in payload: {missing_group}")
        print(f"Joint config min_count for ranking: {max(args.min_count_config, 1)}")
        print(f"Top-N configs per model: {max(args.top_n_configs, 1)}")
        print(f"Saved analysis to: {target_out_dir}")


if __name__ == "__main__":
    main()
