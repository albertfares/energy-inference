import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "energy_cpu_linear.joblib"


def load_model_payload(model_path: Path) -> dict:
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


def select_model_and_features(
    payload: dict,
    *,
    model_name: str,
    model_task_override: str | None,
):
    model = payload.get("model")
    feature_names = payload.get("feature_names")
    group_by = payload.get("group_by")
    models_by_group = payload.get("models_by_group")
    feature_names_by_group = payload.get("feature_names_by_group")
    models_by_task = payload.get("models_by_task")
    feature_names_by_task = payload.get("feature_names_by_task")

    selected_task = (model_task_override or infer_model_task(model_name)).strip().lower()
    selected_model_name = model_name.strip().lower()

    if (
        group_by is not None
        and models_by_group is not None
        and feature_names_by_group is not None
    ):
        key = selected_model_name if group_by == "model" else selected_task
        if key not in models_by_group:
            available = sorted(models_by_group.keys())
            raise ValueError(
                f"{group_by} '{key}' not available in saved model. Available: {available}"
            )
        return models_by_group[key], feature_names_by_group[key]

    if models_by_task is not None and feature_names_by_task is not None:
        if selected_task not in models_by_task:
            available = sorted(models_by_task.keys())
            raise ValueError(f"task '{selected_task}' not available. Available: {available}")
        return models_by_task[selected_task], feature_names_by_task[selected_task]

    if model is None or feature_names is None:
        raise ValueError("Saved payload missing model or feature_names.")
    return model, feature_names


def predict_energy_cpu_j(
    model,
    feature_names: list[str],
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
        elif feature.startswith("precision_"):
            row[feature] = 1.0 if feature == precision_key else 0.0

    x = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0.0)
    y_pred = float(model.predict(x)[0])
    if target_transform == "log1p":
        y_pred = math.expm1(y_pred)
    return max(y_pred, 0.0)


def _prompt_text(prompt: str, *, allow_empty: bool = False) -> str:
    while True:
        value = input(prompt).strip()
        if value or allow_empty:
            return value
        print("Please enter a value.")


def _prompt_int(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def _prompt_float(prompt: str) -> float:
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive terminal pipeline to test saved energy predictors."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to saved predictor payload (.joblib).",
    )
    args = parser.parse_args()

    try:
        payload = load_model_payload(Path(args.model_path))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    target = str(payload.get("target", "energy_cpu_J"))
    target_transform = str(payload.get("target_transform", "none"))
    group_by = payload.get("group_by", "single")
    print(f"Loaded predictor: {args.model_path}")
    print(f"target={target}, target_transform={target_transform}, grouping={group_by}")
    print("Type 'q' at the model prompt to exit.")

    while True:
        model_name = _prompt_text("\nModel name: ")
        if model_name.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        flops_total = _prompt_float("FLOPs total: ")
        batch = _prompt_int("Batch size: ")
        resolution = _prompt_int("Resolution (square side): ")
        precision = _prompt_text("Precision [fp32/fp16/bf16]: ")
        model_task_override = _prompt_text(
            "Model task override [classification/detection, Enter for auto]: ",
            allow_empty=True,
        )
        if model_task_override == "":
            model_task_override = None

        try:
            selected_model, selected_feature_names = select_model_and_features(
                payload,
                model_name=model_name,
                model_task_override=model_task_override,
            )
            pred = predict_energy_cpu_j(
                selected_model,
                selected_feature_names,
                flops_total=flops_total,
                batch=batch,
                resolution=resolution,
                model_name=model_name,
                precision=precision,
                target_transform=target_transform,
            )
            print(f"Predicted {target}: {pred:.6f} J")
        except Exception as exc:  # noqa: BLE001
            print(f"Prediction failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
