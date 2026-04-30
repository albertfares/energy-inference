"""
Energy predictor for the camera benchmark pipeline.

Usage::

    from camera_bench.predictor import load_predictor, predict_energy

    pred = load_predictor()          # loads models/camera_energy_predictor.pkl
    result = predict_energy(pred,
                            model="yolov8n",
                            imgsz=640,
                            precision="fp32",
                            actual_fps=30.0)
    print(result)
    # {'energy_per_frame_j': 0.521, 'energy_per_frame_mj': 521.0,
    #  'mean_power_w': 15.6, 'estimator': 'rf'}

The predictor is trained by ``scripts/train_camera_predictor.py``.
Supported models: yolov8n, ssdlite320 (alias for ssdlite320_mobilenet_v3_large).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_PKL = Path(__file__).resolve().parents[2] / "models" / "camera_energy_predictor.pkl"

# Map long model names → short labels used during training
_MODEL_LABELS: dict[str, str] = {
    "yolov8n": "yolov8n",
    "yolov8s": "yolov8n",   # not in training data; fall back to yolov8n
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
    "ssdlite320": "ssdlite320",
    "ssdlite": "ssdlite320",
}

# Known natural FPS ceilings from the benchmark (used to clamp fps input)
_FPS_CEILINGS: dict[str, float] = {
    "yolov8n": 30.2,
    "ssdlite320": 12.9,
}


def load_predictor(path: str | Path | None = None) -> dict[str, Any]:
    """Load the trained predictor payload from disk.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file produced by ``train_camera_predictor.py``.
        Defaults to ``models/camera_energy_predictor.pkl`` relative to the
        project root.

    Returns
    -------
    dict
        The raw joblib payload (``models``, ``feature_names``, …).
    """
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required: pip install joblib") from exc

    pkl = Path(path) if path is not None else _DEFAULT_PKL
    if not pkl.exists():
        raise FileNotFoundError(
            f"Predictor not found at {pkl}.\n"
            "Run:  python scripts/train_camera_predictor.py"
        )
    return joblib.load(pkl)


def predict_energy(
    predictor: dict[str, Any],
    model: str,
    imgsz: int,
    precision: str,
    actual_fps: float,
    *,
    estimator: str | None = None,
) -> dict[str, float]:
    """Predict energy and power for one model configuration.

    Parameters
    ----------
    predictor:
        Payload from :func:`load_predictor`.
    model:
        Model name, e.g. ``"yolov8n"`` or ``"ssdlite320_mobilenet_v3_large"``.
    imgsz:
        Inference resolution (e.g. 320 or 640).
    precision:
        ``"fp32"`` or ``"fp16"``.
    actual_fps:
        Measured or expected frame rate.  Values above the known hardware
        ceiling for this model are silently clamped.
    estimator:
        Force a specific estimator (``"poly2"`` or ``"rf"``).
        Defaults to the best estimator chosen during training.

    Returns
    -------
    dict with keys:
        ``energy_per_frame_j``, ``energy_per_frame_mj``, ``mean_power_w``,
        ``estimator`` (name used).
    """
    short_label = _MODEL_LABELS.get(model.lower())
    if short_label is None:
        raise ValueError(
            f"Unknown model {model!r}. Supported: {sorted(_MODEL_LABELS)}"
        )

    # Build feature row
    feature_names: list[str] = predictor["feature_names"]
    row: dict[str, float] = {name: 0.0 for name in feature_names}

    model_col = f"model_{short_label}"
    if model_col in row:
        row[model_col] = 1.0

    row["imgsz"] = float(imgsz)
    row["is_fp16"] = 1.0 if precision.lower() == "fp16" else 0.0

    # Clamp FPS to the known ceiling
    fps_ceil = _FPS_CEILINGS.get(short_label, 30.0)
    row["actual_fps"] = float(min(actual_fps, fps_ceil))

    X = pd.DataFrame([row])[feature_names]

    est_name = estimator if estimator is not None else predictor["best_estimator"]
    models_by_target = predictor["models"][est_name]

    result: dict[str, float] = {"estimator": est_name}
    for target in predictor["targets"]:
        m = models_by_target[target]
        val = float(np.maximum(m.predict(X)[0], 0.0))
        result[target] = val
        if target == "energy_per_frame_j":
            result["energy_per_frame_mj"] = val * 1000.0

    return result


def predict_batch(
    predictor: dict[str, Any],
    configs: list[dict],
    *,
    estimator: str | None = None,
) -> pd.DataFrame:
    """Predict energy for a list of configuration dicts.

    Each dict must have keys: ``model``, ``imgsz``, ``precision``, ``actual_fps``.

    Returns a DataFrame with one row per config plus all prediction columns.
    """
    rows = []
    for cfg in configs:
        pred = predict_energy(
            predictor,
            model=cfg["model"],
            imgsz=cfg["imgsz"],
            precision=cfg["precision"],
            actual_fps=cfg["actual_fps"],
            estimator=estimator,
        )
        rows.append({**cfg, **pred})
    return pd.DataFrame(rows)
