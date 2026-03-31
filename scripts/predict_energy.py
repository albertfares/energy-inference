import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "energy_cpu_linear.joblib"
DEFAULT_TARGETS = ["energy_cpu_J", "energy_gpu_J", "energy_io_J", "energy_total_J"]


def load_payload(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    payload = load(model_path)
    if not isinstance(payload, dict):
        raise ValueError("Loaded model payload is not a dict.")
    return payload


def infer_model_task(model_name: str) -> str:
    name = model_name.lower()
    if "ssd" in name or "yolo" in name or "rcnn" in name:
        return "detection"
    return "classification"


def _select_model_and_features(
    payload: dict,
    *,
    target_name: str,
    selected_model_name: str,
    selected_task: str,
):
    models_by_group_by_target = payload.get("models_by_group_by_target")
    feature_names_by_group_by_target = payload.get("feature_names_by_group_by_target")
    group_by = payload.get("group_by")

    # New multi-target format.
    if (
        isinstance(models_by_group_by_target, dict)
        and isinstance(feature_names_by_group_by_target, dict)
    ):
        models_by_group = models_by_group_by_target.get(target_name)
        feature_names_by_group = feature_names_by_group_by_target.get(target_name)
        if not isinstance(models_by_group, dict) or not isinstance(feature_names_by_group, dict):
            return None, None
        key = selected_model_name if group_by == "model" else selected_task
        if key not in models_by_group or key not in feature_names_by_group:
            return None, None
        return models_by_group[key], feature_names_by_group[key]

    # Backward-compatible single-target format.
    models_by_group = payload.get("models_by_group")
    feature_names_by_group = payload.get("feature_names_by_group")
    models_by_task = payload.get("models_by_task")
    feature_names_by_task = payload.get("feature_names_by_task")
    model_single = payload.get("model")
    feature_names_single = payload.get("feature_names")

    if (
        group_by is not None
        and isinstance(models_by_group, dict)
        and isinstance(feature_names_by_group, dict)
    ):
        key = selected_model_name if group_by == "model" else selected_task
        if key not in models_by_group or key not in feature_names_by_group:
            return None, None
        return models_by_group[key], feature_names_by_group[key]

    if isinstance(models_by_task, dict) and isinstance(feature_names_by_task, dict):
        if selected_task not in models_by_task or selected_task not in feature_names_by_task:
            return None, None
        return models_by_task[selected_task], feature_names_by_task[selected_task]

    if model_single is not None and feature_names_single is not None:
        return model_single, feature_names_single
    return None, None


def predict_single_target(
    model,
    feature_names,
    *,
    flops_total: float,
    batch: int,
    resolution: int,
    model_name: str,
    precision: str,
    target_transform: str,
) -> float:
    row = {
        "flops_total": flops_total,
        "batch": batch,
        "resolution": resolution,
    }
    model_key = f"model_{model_name.lower().strip()}"
    precision_key = f"precision_{precision.lower().strip()}"
    for feature in feature_names:
        if feature.startswith("model_"):
            row[feature] = 1.0 if feature == model_key else 0.0
        if feature.startswith("precision_"):
            row[feature] = 1.0 if feature == precision_key else 0.0

    data = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0.0)
    y_pred = model.predict(data)
    value = float(y_pred[0])
    if target_transform == "log1p":
        value = math.expm1(value)
    return max(value, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict energy targets (CPU/GPU/IO/total) from static features: "
            "model, flops_total, batch, resolution, precision."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a saved Joblib model (default: results/models/energy_cpu_linear.joblib).",
    )
    parser.add_argument(
        "--flops-total",
        type=float,
        required=True,
        help="Total FLOPs for the configuration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., resnet18, vit_b_16, yolov8n).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Batch size.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Input resolution (short side, e.g., 224, 320, 640).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision label (e.g., fp32, fp16, bf16).",
    )
    parser.add_argument(
        "--model-task",
        type=str,
        default=None,
        choices=["classification", "detection"],
        help="Optional override for task-specific model selection.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help=(
            "Optional subset of targets to predict. "
            "Defaults to all available energy targets in payload."
        ),
    )

    args = parser.parse_args()

    try:
        payload = load_payload(Path(args.model_path))
        selected_task = (args.model_task or infer_model_task(args.model)).strip().lower()
        selected_model_name = args.model.strip().lower()
        available_targets = payload.get("targets")
        if not isinstance(available_targets, list) or not available_targets:
            available_targets = [str(payload.get("target", "energy_cpu_J"))]
        available_targets = [str(t) for t in available_targets]

        if args.targets:
            requested_targets = [str(t).strip() for t in args.targets if str(t).strip()]
            invalid_targets = [t for t in requested_targets if t not in available_targets]
            if invalid_targets:
                raise ValueError(
                    f"Requested targets not found in payload: {invalid_targets}. "
                    f"Available: {available_targets}"
                )
            targets_to_predict = requested_targets
        else:
            targets_to_predict = [t for t in DEFAULT_TARGETS if t in available_targets]
            if not targets_to_predict:
                targets_to_predict = available_targets

        target_transforms = payload.get("target_transforms")
        if not isinstance(target_transforms, dict):
            default_transform = str(payload.get("target_transform", "none"))
            target_transforms = {t: default_transform for t in available_targets}

        for target_name in targets_to_predict:
            selected_model, selected_feature_names = _select_model_and_features(
                payload,
                target_name=target_name,
                selected_model_name=selected_model_name,
                selected_task=selected_task,
            )
            if selected_model is None or selected_feature_names is None:
                print(
                    f"Skipping {target_name}: no compatible grouped model for "
                    f"model='{selected_model_name}', task='{selected_task}'."
                )
                continue

            transform = str(target_transforms.get(target_name, "none"))
            y_pred = predict_single_target(
                selected_model,
                selected_feature_names,
                flops_total=args.flops_total,
                batch=args.batch,
                resolution=args.resolution,
                model_name=args.model,
                precision=args.precision,
                target_transform=transform,
            )
            print(f"Predicted {target_name}: {y_pred:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error while running prediction: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

