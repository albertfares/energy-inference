"""
Train the *independent-brick* decomposed energy/FPS predictor (Phase 4).

Difference from train_decomposed_predictor.py (Phase 3)
-------------------------------------------------------
Phase 3 still fit every sub-predictor on the SAME joint full-pipeline sweep, so
the bricks were architecturally separate (per-model dicts) but not *data*
independent — capture energy was measured with a model resident, YOLO's
preprocess was fused into inference, etc.

Phase 4 fixes that. Each brick is trained on its OWN isolated benchmark, exactly
as the supervisor asked — a lego block that knows nothing about the others:

    brick        trained from                       feature(s)
    ----------   --------------------------------   ----------------------------
    capture      scripts/bench_capture.py           resolution, target_fps
    preprocess   scripts/bench_preprocess.py        in_pixels, out_pixels, fp16
    infer        scripts/bench_inference.py         imgsz, fp16   (PER MODEL)
    postprocess  scripts/bench_postprocess.py       n_boxes  ("number of objects")

The combination layer stitches them:

    T_other   = T_pre + T_infer + T_post
    T_capture = camera-limited?  camera_period - T_other   : T_decode      (Phase 2 logic)
    T_compute = T_capture + T_pre + T_infer + T_post
    T_frame   = max(T_compute, 1000/target_fps)            (throttle floor)
    FPS       = 1000 / T_frame
    E_total   = E_capture + E_pre + E_infer + E_post + P_sleep * T_sleep

FPS and E_total remain derived OUTPUTS, never inputs.

Inputs
------
  --capture-csv      capture_bench.csv      (benchmark A)
  --preprocess-csv   preprocess_bench.csv   (benchmark B)
  --inference-csv    inference_bench.csv    (benchmark C)
  --postprocess-csv  postprocess_bench.csv  (benchmark D)

Output
------
  models/decomposed_predictor_v2.pkl
  results/analysis/decomposed_v2_<ts>/brick_fit_quality.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "decomposed_predictor_v2.pkl"

MODEL_SHORT = {
    "yolov8n": "yolov8n",
    "yolov8s": "yolov8s",
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
    "ssdlite320": "ssdlite320",
}

# Default number of candidate boxes presented to NMS at predict time, used when
# the caller doesn't specify one. NMS is a tiny fraction of frame cost, so this
# default barely moves the end-to-end numbers; it is exposed for completeness.
DEFAULT_N_CANDIDATE_BOXES = 100.0


# ── Model builder ──────────────────────────────────────────────────────────────

def make_model(degree: int = 2) -> Pipeline:
    """Polynomial ridge — captures the mild nonlinearities (incl. NMS O(N²))."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge",  Ridge(alpha=1.0)),
    ])


def _fit_target(X: pd.DataFrame, y: pd.Series, degree: int = 2):
    """
    Fit one (latency or energy) target. Returns a marker:
      None                 — target all-NaN
      ("constant", value)  — target constant or no usable feature variation
      Pipeline             — fitted regressor
    """
    if y.isna().all():
        return None
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return None
    if y.std() < 1e-12 or all(X[c].std() < 1e-12 for c in X.columns):
        return ("constant", float(y.mean()))
    eff_degree = degree if len(y) > 3 else 1
    m = make_model(eff_degree)
    m.fit(X, y)
    return m


def _predict_marker(marker, X: pd.DataFrame) -> float:
    if marker is None:
        return 0.0
    if isinstance(marker, tuple) and marker[0] == "constant":
        return float(marker[1])
    return float(np.maximum(marker.predict(X)[0], 0.0))


def _loo_mape(X: pd.DataFrame, y: pd.Series, degree: int = 2) -> float:
    """Leave-one-out MAPE for a small brick grid (diagnostic only)."""
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    n = len(y)
    if n < 3 or y.std() < 1e-12:
        return float("nan")
    errs = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        marker = _fit_target(X.iloc[tr], y.iloc[tr], degree)
        pred = _predict_marker(marker, X.iloc[[i]])
        if abs(y.iloc[i]) > 1e-9:
            errs.append(abs(pred - y.iloc[i]) / abs(y.iloc[i]))
    return float(np.mean(errs) * 100) if errs else float("nan")


# ── Feature builders (must match predict_pipeline's row) ────────────────────────

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["in_pixels"]  = df["in_pixels"].astype(float)
    feat["out_pixels"] = df["out_pixels"].astype(float)
    feat["is_fp16"]    = df["is_fp16"].astype(float)
    return feat


def infer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["imgsz"]   = df["imgsz"].astype(float)
    feat["is_fp16"] = df["is_fp16"].astype(float)
    return feat


def postprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NMS cost drivers. ``n_boxes`` (candidate count, the "number of objects"
    feature) dominates — the pairwise-IoU step is ~O(N²), which the degree-2
    polynomial captures from this single column. ``iou_threshold`` is a real but
    second-order driver: a higher threshold suppresses fewer boxes per pass,
    nudging the survivor count and the inner-loop work.
    """
    feat = pd.DataFrame(index=df.index)
    feat["n_boxes"]       = df["n_boxes"].astype(float)
    feat["iou_threshold"] = pd.to_numeric(df["iou_threshold"], errors="coerce").fillna(0.45)
    return feat


# ── Brick trainers ──────────────────────────────────────────────────────────────

def train_capture_brick(csv_path: Path) -> dict:
    """
    Derive per-resolution camera constants and the sleep power from benchmark A.

      T_camera_period_ms : per resolution, from the target_fps==0 (unbounded) row
      T_decode_ms        : per resolution, median capture latency among throttled
                           rows (frame already buffered → pure decode)
      E_decode_mj        : per-iter energy of those throttled rows
      P_sleep_w          : board power during a heavily-throttled capture run
                           (the read is negligible, so mean power ≈ idle/sleep)
    """
    df = pd.read_csv(csv_path)
    constants: dict = {}
    sleep_powers: list[float] = []

    for (w, h), g in df.groupby(["width", "height"]):
        unbounded = g[g["target_fps"] == 0]
        throttled = g[g["target_fps"] > 0]

        T_period = (float(unbounded["capture_lat_mean_ms"].median())
                    if not unbounded.empty else float("nan"))
        if not throttled.empty:
            T_decode = float(throttled["capture_lat_mean_ms"].median())
            E_decode = float(throttled["capture_energy_mj_per_iter"].median())
            # Highest FPS cap → smallest duty cycle → mean power closest to idle.
            most_throttled = throttled.sort_values("target_fps").iloc[0]
            sleep_powers.append(float(most_throttled["mean_power_w"]))
        else:
            T_decode, E_decode = 0.5, 9.0  # conservative fallback

        constants[(int(w), int(h))] = {
            "T_camera_period_ms": T_period,
            "T_decode_ms": T_decode,
            "E_decode_mj": E_decode,
        }
        period_str = f"{T_period:.2f}" if not np.isnan(T_period) else " n/a"
        print(f"  capture {int(w)}x{int(h)}: period={period_str} ms  "
              f"T_decode={T_decode:.3f} ms  E_decode={E_decode:.3f} mJ")

    p_sleep = float(np.median(sleep_powers)) if sleep_powers else 0.0
    print(f"  sleep power (from throttled capture): {p_sleep:.3f} W")
    return {"camera_constants": constants, "p_sleep_w": p_sleep}


def train_regression_brick(
    csv_path: Path,
    features_fn,
    lat_col: str,
    energy_col: str,
    group_col: str | None,
    label: str,
) -> tuple[dict, list[dict]]:
    """
    Generic regression brick. If group_col is given (e.g. 'model_short'), returns
    a dict keyed by group; otherwise a single {lat,energy} dict.

    Returns (brick, fit_quality_rows).
    """
    df = pd.read_csv(csv_path)
    quality: list[dict] = []

    def fit_subset(sub: pd.DataFrame, tag: str) -> dict:
        X = features_fn(sub)
        lat = _fit_target(X, pd.to_numeric(sub[lat_col], errors="coerce"))
        eng = _fit_target(X, pd.to_numeric(sub[energy_col], errors="coerce"))
        lat_mape = _loo_mape(X, pd.to_numeric(sub[lat_col], errors="coerce"))
        eng_mape = _loo_mape(X, pd.to_numeric(sub[energy_col], errors="coerce"))
        quality.append({
            "brick": label, "group": tag, "n_rows": len(sub),
            "lat_LOO_MAPE_%": round(lat_mape, 2) if lat_mape == lat_mape else None,
            "energy_LOO_MAPE_%": round(eng_mape, 2) if eng_mape == eng_mape else None,
        })
        print(f"  {label}[{tag}] n={len(sub)}  "
              f"lat LOO-MAPE={lat_mape:5.1f}%  energy LOO-MAPE={eng_mape:5.1f}%"
              if lat_mape == lat_mape else
              f"  {label}[{tag}] n={len(sub)}  (LOO n/a)")
        return {"lat": lat, "energy": eng}

    if group_col is None:
        return fit_subset(df, "shared"), quality

    brick: dict = {}
    for gval, sub in df.groupby(group_col):
        brick[str(gval)] = fit_subset(sub, str(gval))
    return brick, quality


# ── Combination layer ────────────────────────────────────────────────────────────

def compute_capture_outputs(
    camera_constants: dict, p_sleep_w: float,
    width: int, height: int, target_fps: float, T_other_ms: float,
) -> tuple[float, float]:
    """Phase 2 capture combination logic (camera-limited vs compute-limited)."""
    cam = camera_constants.get((int(width), int(height)))
    if cam is None:
        cam = next(iter(camera_constants.values()))
    T_decode = float(cam["T_decode_ms"])
    E_decode = float(cam["E_decode_mj"])
    T_period = float(cam["T_camera_period_ms"])

    if target_fps > 0 or np.isnan(T_period):
        T_capture = T_decode
    else:
        T_capture = max(T_decode, T_period - T_other_ms)

    wait_s = max(0.0, T_capture - T_decode) / 1000.0
    E_capture = E_decode + p_sleep_w * wait_s * 1000.0
    return T_capture, E_capture


def predict_pipeline(
    payload: dict,
    model: str,
    imgsz: int,
    precision: str,
    target_fps: float = 0,
    width: int = 640,
    height: int = 480,
    n_candidate_boxes: float | None = None,
    iou_threshold: float = 0.45,
) -> dict:
    """
    Predict FPS and per-stage energy from independent bricks.

    n_candidate_boxes : number of boxes presented to NMS (the postprocess brick's
                        'number of objects' feature). Defaults to
                        DEFAULT_N_CANDIDATE_BOXES.
    iou_threshold     : NMS IoU threshold fed to the postprocess brick.
    """
    bricks   = payload["bricks"]
    cap      = payload["capture"]
    p_sleep  = cap["p_sleep_w"]
    is_fp16  = float(precision.lower() == "fp16")
    model_short = MODEL_SHORT.get(model, model)
    n_boxes = float(n_candidate_boxes if n_candidate_boxes is not None
                    else DEFAULT_N_CANDIDATE_BOXES)

    # ── preprocess (shared brick) ──
    pre_row = pd.DataFrame([{
        "in_pixels": float(width) * float(height),
        "out_pixels": float(imgsz) ** 2,
        "is_fp16": is_fp16,
    }])
    pre = bricks["preprocess"]
    Xpre = preprocess_features(pre_row)
    T_pre = _predict_marker(pre["lat"], Xpre)
    E_pre = _predict_marker(pre["energy"], Xpre)

    # ── inference (per-model brick) ──
    if model_short not in bricks["infer"]:
        raise ValueError(
            f"No inference brick for model '{model_short}'. "
            f"Available: {sorted(bricks['infer'].keys())}. Run "
            f"scripts/bench_inference.py for this model first."
        )
    inf_row = pd.DataFrame([{"imgsz": float(imgsz), "is_fp16": is_fp16}])
    inf = bricks["infer"][model_short]
    Xinf = infer_features(inf_row)
    T_inf = _predict_marker(inf["lat"], Xinf)
    E_inf = _predict_marker(inf["energy"], Xinf)

    # ── postprocess / NMS (shared brick, number-of-objects feature) ──
    post_row = pd.DataFrame([{"n_boxes": n_boxes, "iou_threshold": float(iou_threshold)}])
    post = bricks["postprocess"]
    Xpost = postprocess_features(post_row)
    T_post = _predict_marker(post["lat"], Xpost)
    E_post = _predict_marker(post["energy"], Xpost)

    # ── capture (combination from camera constants) ──
    T_other = T_pre + T_inf + T_post
    T_cap, E_cap = compute_capture_outputs(
        cap["camera_constants"], p_sleep, width, height, target_fps, T_other)

    # ── combine ──
    T_compute = T_cap + T_pre + T_inf + T_post
    if target_fps > 0:
        T_frame = max(T_compute, 1000.0 / target_fps)
    else:
        T_frame = T_compute
    fps = 1000.0 / max(T_frame, 1e-3)

    T_sleep_ms = max(0.0, T_frame - T_compute)
    E_overhead = p_sleep * (T_sleep_ms / 1000.0) * 1000.0
    E_total = E_cap + E_pre + E_inf + E_post + E_overhead

    return {
        "fps": round(fps, 2),
        "T_frame_ms": round(T_frame, 2),
        "E_total_mj": round(E_total, 1),
        "E_capture_mj": round(E_cap, 1), "T_capture_ms": round(T_cap, 2),
        "E_preprocess_mj": round(E_pre, 1), "T_preprocess_ms": round(T_pre, 3),
        "E_infer_mj": round(E_inf, 1), "T_infer_ms": round(T_inf, 3),
        "E_postprocess_mj": round(E_post, 1), "T_postprocess_ms": round(T_post, 4),
        "E_overhead_mj": round(E_overhead, 1),
        "n_candidate_boxes": n_boxes,
        "bottleneck": "throttle" if T_frame > T_compute else "compute",
    }


# ── Main ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    base = PROJECT_ROOT / "results" / "isolated_bench"
    p = argparse.ArgumentParser(description="Train independent-brick decomposed predictor.")
    p.add_argument("--capture-csv", type=Path,
                   default=base / "capture" / "capture_bench.csv")
    p.add_argument("--preprocess-csv", type=Path,
                   default=base / "preprocess" / "preprocess_bench.csv")
    p.add_argument("--inference-csv", type=Path,
                   default=base / "inference" / "inference_bench.csv")
    p.add_argument("--postprocess-csv", type=Path,
                   default=base / "postprocess" / "postprocess_bench.csv")
    p.add_argument("--out", type=Path, default=MODEL_SAVE_PATH)
    return p


def _require(path: Path, bench: str) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            f"       Run {bench} on the Jetson first (see its module docstring)."
        )


def main() -> None:
    args = build_parser().parse_args()
    _require(args.capture_csv,     "scripts/bench_capture.py")
    _require(args.preprocess_csv,  "scripts/bench_preprocess.py")
    _require(args.inference_csv,   "scripts/bench_inference.py")
    _require(args.postprocess_csv, "scripts/bench_postprocess.py")

    print("Training capture brick ...")
    capture = train_capture_brick(args.capture_csv)

    quality: list[dict] = []
    print("\nTraining preprocess brick (shared) ...")
    pre_brick, q = train_regression_brick(
        args.preprocess_csv, preprocess_features,
        "preprocess_lat_mean_ms", "preprocess_energy_mj_per_iter",
        group_col=None, label="preprocess")
    quality += q

    print("\nTraining inference bricks (per model) ...")
    inf_brick, q = train_regression_brick(
        args.inference_csv, infer_features,
        "infer_lat_mean_ms", "infer_energy_mj_per_iter",
        group_col="model_short", label="infer")
    quality += q

    print("\nTraining postprocess/NMS brick (shared) ...")
    post_brick, q = train_regression_brick(
        args.postprocess_csv, postprocess_features,
        "postprocess_lat_mean_ms", "postprocess_energy_mj_per_iter",
        group_col=None, label="postprocess")
    quality += q

    payload = {
        "version": "phase4_independent_bricks",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "capture": capture,
        "bricks": {
            "preprocess": pre_brick,
            "infer": inf_brick,
            "postprocess": post_brick,
        },
        "model_short": MODEL_SHORT,
        "default_n_candidate_boxes": DEFAULT_N_CANDIDATE_BOXES,
        "sources": {
            "capture": str(args.capture_csv),
            "preprocess": str(args.preprocess_csv),
            "inference": str(args.inference_csv),
            "postprocess": str(args.postprocess_csv),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"\nSaved payload → {args.out}")

    # Fit-quality report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "analysis" / f"decomposed_v2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qdf = pd.DataFrame(quality)
    qdf.to_csv(out_dir / "brick_fit_quality.csv", index=False)
    print(f"Per-brick fit quality → {out_dir / 'brick_fit_quality.csv'}")
    print(qdf.to_string(index=False))

    # Smoke-test predict on the models that have an inference brick
    print("\nSmoke-test predictions (unbounded):")
    for m in inf_brick.keys():
        try:
            out = predict_pipeline(payload, m, imgsz=640, precision="fp32", target_fps=0)
            print(f"  {m}: FPS={out['fps']}  E_total={out['E_total_mj']} mJ  "
                  f"(cap={out['T_capture_ms']} pre={out['T_preprocess_ms']} "
                  f"inf={out['T_infer_ms']} post={out['T_postprocess_ms']} ms)")
        except Exception as exc:
            print(f"  {m}: predict failed — {exc}")


if __name__ == "__main__":
    main()
