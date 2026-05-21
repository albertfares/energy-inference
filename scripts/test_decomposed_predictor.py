"""
Test / interactive runner for the decomposed energy predictor.

Usage examples
--------------
# Predict one config (no FPS cap):
python3 scripts/test_decomposed_predictor.py --model yolov8n --imgsz 640 --precision fp32

# Predict with a target FPS cap:
python3 scripts/test_decomposed_predictor.py --model yolov8n --imgsz 640 --precision fp32 --target-fps 10

# Compare prediction against every measured row in the sweep data:
python3 scripts/test_decomposed_predictor.py --validate

# Validate for one specific config only:
python3 scripts/test_decomposed_predictor.py --validate --model yolov8n --imgsz 640 --precision fp32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PREDICTOR_PATH = PROJECT_ROOT / "models" / "decomposed_predictor.pkl"

SWEEP_DIRS = [
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_133405",
    PROJECT_ROOT / "results" / "camera_bench" / "yolov8n_fps_sweep_MAXN_20260428_150803",
    PROJECT_ROOT / "results" / "camera_bench" / "ssdlite_fps_sweep_MAXN_20260428_162918",
]

MODEL_LABELS = {
    "yolov8n": "yolov8n",
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
    "ssdlite320": "ssdlite320",
    "ssdlite": "ssdlite320",
}


# ── Load predictor ─────────────────────────────────────────────────────────────

def load_predictor() -> dict:
    if not PREDICTOR_PATH.exists():
        print(f"ERROR: predictor not found at {PREDICTOR_PATH}", file=sys.stderr)
        print("Run: python3 scripts/train_decomposed_predictor.py", file=sys.stderr)
        sys.exit(1)
    # The payload contains pickled references to functions defined in
    # train_decomposed_predictor — import it first so joblib can find them.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import train_decomposed_predictor  # noqa: F401
    return joblib.load(PREDICTOR_PATH)


# ── Pretty-print one prediction ───────────────────────────────────────────────

def print_prediction(pred: dict, model: str, imgsz: int, precision: str,
                     target_fps: float, actual: dict | None = None) -> None:
    fps_str = f"{target_fps:.0f}" if target_fps > 0 else "unbounded"
    print(f"\n{'─'*60}")
    print(f"  Config : {model}  imgsz={imgsz}  {precision}  target_fps={fps_str}")
    print(f"{'─'*60}")

    # Stage breakdown table
    stages = [
        ("Capture",     pred["T_capture_ms"],     pred["E_capture_mj"]),
        ("Preprocess",  pred["T_preprocess_ms"],  pred["E_preprocess_mj"]),
        ("Inference",   pred["T_infer_ms"],        pred["E_infer_mj"]),
        ("Postprocess", pred["T_postprocess_ms"],  pred["E_postprocess_mj"]),
        ("Overhead",    None,                      pred["E_overhead_mj"]),
    ]

    print(f"  {'Stage':<14}  {'Latency':>10}  {'Energy/frame':>14}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*14}")
    for name, lat, energy in stages:
        lat_s    = f"{lat:.1f} ms" if lat is not None else "   (∝ T_frame)"
        energy_s = f"{energy:.1f} mJ"
        print(f"  {name:<14}  {lat_s:>10}  {energy_s:>14}")

    print(f"  {'─'*14}  {'─'*10}  {'─'*14}")
    print(f"  {'TOTAL':<14}  {pred['T_frame_ms']:>7.1f} ms  {pred['E_total_mj']:>11.1f} mJ")
    print()
    print(f"  Predicted FPS  : {pred['fps']:.1f}")
    print(f"  Bottleneck     : {pred['bottleneck']}")

    if actual is not None:
        actual_fps = actual.get("fps_mean", float("nan"))
        actual_e   = actual.get("energy_per_frame_j", float("nan")) * 1000

        fps_err  = (pred["fps"]         - actual_fps) / max(actual_fps, 1e-6) * 100
        e_err    = (pred["E_total_mj"]  - actual_e)   / max(actual_e,   1e-6) * 100

        print()
        print(f"  {'':14}  {'Predicted':>10}  {'Actual':>10}  {'Error':>8}")
        print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*8}")
        print(f"  {'FPS':<14}  {pred['fps']:>10.1f}  {actual_fps:>10.1f}  {fps_err:>+7.1f}%")
        print(f"  {'E/frame (mJ)':<14}  {pred['E_total_mj']:>10.1f}  {actual_e:>10.1f}  {e_err:>+7.1f}%")

    print(f"{'─'*60}\n")


# ── Load sweep data for validation ────────────────────────────────────────────

def load_sweep_data() -> pd.DataFrame:
    frames = []
    for d in SWEEP_DIRS:
        csv = d / "sweep_summary.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "yolo_imgsz" not in df.columns:
            df["yolo_imgsz"] = 640
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["model_short"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["imgsz"] = pd.to_numeric(df["yolo_imgsz"], errors="coerce").fillna(640).astype(int)
    return df


# ── Validate against all measured rows ────────────────────────────────────────

def run_validation(payload: dict, filter_model: str | None,
                   filter_imgsz: int | None, filter_prec: str | None) -> None:
    from train_decomposed_predictor import predict_pipeline

    df = load_sweep_data()

    # Apply filters
    if filter_model:
        short = MODEL_LABELS.get(filter_model, filter_model)
        df = df[df["model_short"] == short]
    if filter_imgsz:
        df = df[df["imgsz"] == filter_imgsz]
    if filter_prec:
        df = df[df["precision"].str.lower() == filter_prec.lower()]

    if df.empty:
        print("No matching rows found in sweep data.")
        return

    print(f"\nValidating {len(df)} rows...\n")

    pred_fps_list, actual_fps_list = [], []
    pred_e_list,   actual_e_list   = [], []

    rows_out = []
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
        actual_fps = float(row["fps_mean"])
        actual_e   = float(row["energy_per_frame_j"]) * 1000

        pred_fps_list.append(pred["fps"])
        actual_fps_list.append(actual_fps)
        pred_e_list.append(pred["E_total_mj"])
        actual_e_list.append(actual_e)

        rows_out.append({
            "model":       row["model_short"],
            "imgsz":       row["imgsz"],
            "precision":   row["precision"],
            "target_fps":  row["target_fps"],
            "pred_fps":    round(pred["fps"], 1),
            "actual_fps":  round(actual_fps, 1),
            "fps_err_%":   round((pred["fps"] - actual_fps) / max(actual_fps, 1e-6) * 100, 1),
            "pred_E_mj":   round(pred["E_total_mj"], 1),
            "actual_E_mj": round(actual_e, 1),
            "E_err_%":     round((pred["E_total_mj"] - actual_e) / max(actual_e, 1e-6) * 100, 1),
            "bottleneck":  pred["bottleneck"],
        })

    results = pd.DataFrame(rows_out)

    # ── Per-config summary ────────────────────────────────────────────────────
    print(f"  {'Config':<40}  {'FPS MAPE':>9}  {'E MAPE':>9}  {'FPS MAE':>9}  {'E MAE':>10}")
    print(f"  {'─'*40}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*10}")

    for cfg, grp in results.groupby(["model","imgsz","precision"]):
        fps_mape = np.mean(np.abs(grp["fps_err_%"]))
        e_mape   = np.mean(np.abs(grp["E_err_%"]))
        fps_mae  = np.mean(np.abs(grp["pred_fps"] - grp["actual_fps"]))
        e_mae    = np.mean(np.abs(grp["pred_E_mj"] - grp["actual_E_mj"]))
        label    = f"{cfg[0]}_imgsz{cfg[1]}_{cfg[2]}"
        print(f"  {label:<40}  {fps_mape:>8.1f}%  {e_mape:>8.1f}%  "
              f"{fps_mae:>7.1f}fps  {e_mae:>8.1f}mJ")

    # ── Overall summary ───────────────────────────────────────────────────────
    pred_fps_arr   = np.array(pred_fps_list)
    actual_fps_arr = np.array(actual_fps_list)
    pred_e_arr     = np.array(pred_e_list)
    actual_e_arr   = np.array(actual_e_list)

    fps_mape_all = float(np.mean(np.abs(pred_fps_arr - actual_fps_arr) / np.maximum(actual_fps_arr, 1e-6)) * 100)
    e_mape_all   = float(np.mean(np.abs(pred_e_arr   - actual_e_arr)   / np.maximum(actual_e_arr,   1e-6)) * 100)
    fps_mae_all  = float(np.mean(np.abs(pred_fps_arr - actual_fps_arr)))
    e_mae_all    = float(np.mean(np.abs(pred_e_arr   - actual_e_arr)))

    print(f"\n  {'OVERALL':<40}  {fps_mape_all:>8.1f}%  {e_mape_all:>8.1f}%  "
          f"{fps_mae_all:>7.1f}fps  {e_mae_all:>8.1f}mJ")

    # ── Detailed table ────────────────────────────────────────────────────────
    print(f"\n{'─'*100}")
    print("Full results per row:")
    print(results.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the decomposed energy predictor")
    p.add_argument("--model",      default="yolov8n",
                   help="Model name: yolov8n | ssdlite320  (default: yolov8n)")
    p.add_argument("--imgsz",      type=int, default=640,
                   help="Model input size (default: 640)")
    p.add_argument("--precision",  default="fp32",
                   help="fp32 | fp16  (default: fp32)")
    p.add_argument("--target-fps", type=float, default=0,
                   help="FPS cap; 0 = unbounded  (default: 0)")
    p.add_argument("--width",      type=int, default=640,
                   help="Camera width  (default: 640)")
    p.add_argument("--height",     type=int, default=480,
                   help="Camera height (default: 480)")
    p.add_argument("--validate",   action="store_true",
                   help="Compare predictions against all measured sweep rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_predictor()

    from train_decomposed_predictor import predict_pipeline

    if args.validate:
        filter_model = None if args.model == "yolov8n" and not any(
            a.startswith("--model") for a in sys.argv[1:]
        ) else args.model
        # Re-parse to detect if --model was explicitly passed
        explicit_model = "--model" in sys.argv
        explicit_imgsz = "--imgsz" in sys.argv
        explicit_prec  = "--precision" in sys.argv
        run_validation(
            payload,
            filter_model = args.model if explicit_model else None,
            filter_imgsz = args.imgsz if explicit_imgsz else None,
            filter_prec  = args.precision if explicit_prec else None,
        )
    else:
        pred = predict_pipeline(
            payload,
            model      = args.model,
            imgsz      = args.imgsz,
            precision  = args.precision,
            target_fps = args.target_fps,
            width      = args.width,
            height     = args.height,
        )

        # Try to find a matching measured row to compare against
        actual = None
        try:
            df = load_sweep_data()
            short = MODEL_LABELS.get(args.model, args.model)
            match = df[
                (df["model_short"] == short) &
                (df["imgsz"] == args.imgsz) &
                (df["precision"].str.lower() == args.precision.lower()) &
                (df["target_fps"] == args.target_fps)
            ]
            if not match.empty:
                actual = match.iloc[0].to_dict()
                if len(match) > 1:
                    # Use median over repeats
                    actual = {
                        "fps_mean":            match["fps_mean"].median(),
                        "energy_per_frame_j":  match["energy_per_frame_j"].median(),
                    }
        except Exception:
            pass

        print_prediction(pred, args.model, args.imgsz, args.precision,
                         args.target_fps, actual)


if __name__ == "__main__":
    main()
