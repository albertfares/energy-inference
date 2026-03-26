import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "energy_cpu_linear.joblib"


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    payload = load(model_path)
    model = payload.get("model")
    feature_names = payload.get("feature_names")
    group_by = payload.get("group_by")
    models_by_group = payload.get("models_by_group")
    feature_names_by_group = payload.get("feature_names_by_group")
    models_by_task = payload.get("models_by_task")
    feature_names_by_task = payload.get("feature_names_by_task")
    target = payload.get("target", "energy_cpu_J")
    target_transform = payload.get("target_transform", "none")
    return (
        model,
        feature_names,
        group_by,
        models_by_group,
        feature_names_by_group,
        models_by_task,
        feature_names_by_task,
        target,
        target_transform,
    )


def infer_model_task(model_name: str) -> str:
    name = model_name.lower()
    if "ssd" in name or "yolo" in name or "rcnn" in name:
        return "detection"
    return "classification"


def predict_single(
    model,
    feature_names,
    *,
    flops_total: float,
    batch: int,
    resolution: int,
    model_name: str,
    precision: str,
    target_transform: str = "none",
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
        import math

        value = math.expm1(value)
    return max(value, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict CPU energy (J) using a trained linear model and static "
            "features: model, flops_total, batch, resolution, precision."
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

    args = parser.parse_args()

    try:
        (
            model,
            feature_names,
            group_by,
            models_by_group,
            feature_names_by_group,
            models_by_task,
            feature_names_by_task,
            target,
            target_transform,
        ) = load_model(Path(args.model_path))

        selected_task = (args.model_task or infer_model_task(args.model)).strip().lower()
        selected_model_name = args.model.strip().lower()

        if (
            group_by is not None
            and models_by_group is not None
            and feature_names_by_group is not None
        ):
            if group_by == "model":
                key = selected_model_name
            else:
                key = selected_task
            if key not in models_by_group:
                raise ValueError(
                    f"{group_by} '{key}' is not available in the saved model. "
                    f"Available: {sorted(models_by_group.keys())}"
                )
            selected_model = models_by_group[key]
            selected_feature_names = feature_names_by_group[key]
        elif models_by_task is not None and feature_names_by_task is not None:
            if selected_task not in models_by_task:
                raise ValueError(
                    f"Task '{selected_task}' is not available in the saved model. "
                    f"Available: {sorted(models_by_task.keys())}"
                )
            selected_model = models_by_task[selected_task]
            selected_feature_names = feature_names_by_task[selected_task]
        else:
            selected_model = model
            selected_feature_names = feature_names

        if selected_model is None or selected_feature_names is None:
            raise ValueError("Loaded model payload is missing model or feature schema.")

        y_pred = predict_single(
            selected_model,
            selected_feature_names,
            flops_total=args.flops_total,
            batch=args.batch,
            resolution=args.resolution,
            model_name=args.model,
            precision=args.precision,
            target_transform=target_transform,
        )
        print(f"Predicted {target}: {y_pred:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error while running prediction: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

