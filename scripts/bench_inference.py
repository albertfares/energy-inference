"""
Benchmark C — isolated INFERENCE stage (synthetic tensor, per model).

Why this exists
---------------
Inference is the one stage that genuinely depends on the model, so it MUST be
benchmarked per model. But it should depend on nothing else: no camera, no real
preprocessing, no NMS. We therefore feed a synthetic input tensor of the right
shape straight into the network's forward pass and time only that.

Backend handling
----------------
  * torchvision detectors: ``model([chw_tensor])`` under ``no_grad``. The model's
    internal transform (normalise / resize to its fixed input) is part of the
    forward and is intentionally included.
  * ultralytics YOLO: we bypass ``predict()`` (which fuses pre+infer+post) and
    call the raw ``model.model(nchw_tensor)`` so only the network forward is
    measured — the pure inference brick, with NMS handled by benchmark D.

Sweep axes
----------
  * model      (yolov8n, ssdlite320_mobilenet_v3_large, ...)
  * imgsz      (network input side)
  * precision  (fp32 / fp16)

Output
------
results/isolated_bench/inference/<run>/inference_bench.csv
    one row per (model, imgsz, precision).

Run on the Jetson::

    PYTHONPATH=src python scripts/bench_inference.py \
        --models yolov8n ssdlite320_mobilenet_v3_large \
        --imgsz 320 640 --precisions fp32 fp16 --n-measure 300
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camera_bench.isolated import (
    IsolatedCSV,
    make_power_monitor,
    measure_loop,
    torch_cuda_sync_fn,
)
from camera_bench.models import SUPPORTED_MODELS, load_detector

MODEL_SHORT = {
    "yolov8n": "yolov8n",
    "yolov8s": "yolov8s",
    "ssdlite320_mobilenet_v3_large": "ssdlite320",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated inference-stage benchmark.")
    p.add_argument("--models", nargs="+", default=["yolov8n", "ssdlite320_mobilenet_v3_large"],
                   choices=SUPPORTED_MODELS)
    p.add_argument("--imgsz", nargs="+", type=int, default=[320, 640])
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16"],
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--n-warmup", type=int, default=30)
    p.add_argument("--n-measure", type=int, default=300)
    p.add_argument("--cooldown-s", type=float, default=5.0)
    p.add_argument("--no-energy", dest="enable_energy", action="store_false")
    p.add_argument("--ina-hz", type=int, default=1000)
    p.add_argument("--ina-hw", default="all", choices=["cpu", "gpu", "io", "both", "all"])
    p.add_argument("--sampler-exe", default="src/energy_inference/tools/sample_ina3221")
    p.add_argument("--out-dir", default="results/isolated_bench/inference")
    p.add_argument("--run-name", default="auto")
    return p


def make_forward(detector: dict, device, imgsz: int, precision: str):
    """Return (forward_callable, note) that runs ONE pure forward pass."""
    import torch

    dtype = torch.float16 if precision == "fp16" else (
        torch.bfloat16 if precision == "bf16" else torch.float32)

    if detector["backend"] == "torchvision":
        model = detector["model"]
        img = torch.rand(3, imgsz, imgsz, device=device)
        if precision == "fp16":
            img = img.half()
        elif precision == "bf16":
            img = img.to(torch.bfloat16)

        def forward() -> None:
            with torch.no_grad():
                model([img])

        return forward, "torchvision model([chw])"

    # ultralytics — raw network forward only (no pre/post/NMS)
    yolo = detector["model"]
    net = yolo.model
    try:
        net.fuse()  # conv+bn fuse BEFORE casting, per models.py guidance
    except Exception:
        pass
    net.eval()
    if precision == "fp16" and device.type == "cuda":
        net.half()
    elif precision == "bf16":
        net.to(torch.bfloat16)
    im = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    def forward() -> None:
        with torch.no_grad():
            net(im)

    return forward, "ultralytics raw model.model(nchw)"


def main() -> None:
    import time
    import torch

    args = build_parser().parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    sync_fn = torch_cuda_sync_fn(device)
    print(f"Torch device: {device}")

    run_name = args.run_name
    if run_name == "auto":
        run_name = "inference_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = IsolatedCSV(out_dir / "inference_bench.csv")
    print(f"Results: {out_dir}")

    for model_name in args.models:
        short = MODEL_SHORT.get(model_name, model_name)
        for prec in args.precisions:
            if prec == "fp16" and device.type != "cuda":
                print(f"  skip {model_name} fp16 on CPU")
                continue
            # Load the model fresh per precision (torchvision casts at load).
            detector = load_detector(model_name, device, prec)
            for imgsz in args.imgsz:
                forward, note = make_forward(detector, device, imgsz, prec)
                pm = make_power_monitor(
                    args.enable_energy, args.sampler_exe,
                    csv_path=str(out_dir / f"power_{short}_{imgsz}_{prec}.csv"),
                    hz=args.ina_hz, hw=args.ina_hw,
                )
                meas = measure_loop(
                    iter_fn=forward,
                    n_warmup=args.n_warmup,
                    n_measure=args.n_measure,
                    power_monitor=pm,
                    sync_fn=sync_fn,
                )
                row = {
                    "model": model_name,
                    "model_short": short,
                    "backend": detector["backend"],
                    "imgsz": imgsz,
                    "precision": prec,
                    "is_fp16": int(prec == "fp16"),
                    "device": device.type,
                    "note": note,
                }
                row.update(meas.as_row("infer"))
                csv.append(row)
                csv.flush()
                print(
                    f"  {short:>10} imgsz={imgsz} {prec}:  "
                    f"T={meas.latency_ms['mean']:8.3f} ms  "
                    f"E/iter={meas.energy_mj_per_iter:8.3f} mJ"
                )
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)
            del detector
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Wrote {len(csv)} rows → {csv.path}")


if __name__ == "__main__":
    main()
