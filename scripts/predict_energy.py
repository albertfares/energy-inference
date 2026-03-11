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
    model = payload["model"]
    feature_names = payload["feature_names"]
    target = payload.get("target", "energy_cpu_J")
    return model, feature_names, target


def predict_single(
    model,
    feature_names,
    *,
    flops_total: float,
    batch: int,
    resolution: int,
    precision: str,
) -> float:
    row = {
        "flops_total": flops_total,
        "batch": batch,
        "resolution": resolution,
    }
    precision_key = f"precision_{precision.lower().strip()}"
    for feature in feature_names:
        if feature.startswith("precision_"):
            row[feature] = 1.0 if feature == precision_key else 0.0

    data = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0.0)
    y_pred = model.predict(data)
    return float(y_pred[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict CPU energy (J) using a trained linear model and static "
            "features: flops_total, batch, resolution, precision."
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

    args = parser.parse_args()

    try:
        model, feature_names, target = load_model(Path(args.model_path))
        y_pred = predict_single(
            model,
            feature_names,
            flops_total=args.flops_total,
            batch=args.batch,
            resolution=args.resolution,
            precision=args.precision,
        )
        print(f"Predicted {target}: {y_pred:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error while running prediction: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

