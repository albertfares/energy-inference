"""
Benchmark B — isolated PREPROCESS stage (synthetic frame, no camera, no model).

Why this exists
---------------
Preprocess turns a raw BGR frame into a model-ready tensor: colour convert,
(optional) resize to the network input size, HWC→CHW, normalise to [0,1], cast
to the working precision, and copy host→device. None of that depends on which
detector runs next, so it should be measured on its own.

This mirrors the torchvision preprocess path in
``camera_bench.detection._run_torchvision_staged`` (cvtColor → from_numpy →
permute → /255 → cast → .to(device)), with an optional resize step so the brick
can also cover pipelines that resize on the CPU before the model.

Sweep axes
----------
  * source resolution   (raw frame WxH the camera produced)
  * destination imgsz    (network input side; controls out_pixels)
  * precision            (fp32 / fp16)

Output
------
results/isolated_bench/preprocess/<run>/preprocess_bench.csv
    one row per (src_w, src_h, dst_imgsz, precision).

Run on the Jetson::

    PYTHONPATH=src python scripts/bench_preprocess.py \
        --src-resolutions 640x480 1280x720 \
        --dst-imgsz 320 640 --precisions fp32 fp16 \
        --resize --n-measure 500
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated preprocess-stage benchmark.")
    p.add_argument("--src-resolutions", nargs="+", default=["640x480"])
    p.add_argument("--dst-imgsz", nargs="+", type=int, default=[320, 640])
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16"],
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--resize", action="store_true",
                   help="Include a CPU resize to dst_imgsz (letterbox-free).")
    p.add_argument("--cpu", action="store_true", help="Force CPU (no H2D copy).")
    p.add_argument("--n-warmup", type=int, default=50)
    p.add_argument("--n-measure", type=int, default=500)
    p.add_argument("--cooldown-s", type=float, default=3.0)
    p.add_argument("--no-energy", dest="enable_energy", action="store_false")
    p.add_argument("--ina-hz", type=int, default=1000)
    p.add_argument("--ina-hw", default="all", choices=["cpu", "gpu", "io", "both", "all"])
    p.add_argument("--sampler-exe", default="src/energy_inference/tools/sample_ina3221")
    p.add_argument("--out-dir", default="results/isolated_bench/preprocess")
    p.add_argument("--run-name", default="auto")
    return p


def main() -> None:
    import time
    import cv2  # type: ignore
    import torch

    args = build_parser().parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    sync_fn = torch_cuda_sync_fn(device)
    print(f"Torch device: {device}")

    run_name = args.run_name
    if run_name == "auto":
        run_name = "preprocess_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = IsolatedCSV(out_dir / "preprocess_bench.csv")
    print(f"Results: {out_dir}")

    src_resolutions = [parse_resolution(r) for r in args.src_resolutions]

    for (sw, sh) in src_resolutions:
        # Fixed synthetic frame for this source resolution (random but stable).
        rng = np.random.default_rng(0)
        frame_bgr = rng.integers(0, 256, size=(sh, sw, 3), dtype=np.uint8)

        for imgsz in args.dst_imgsz:
            for prec in args.precisions:
                def preprocess_one() -> None:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if args.resize:
                        rgb = cv2.resize(rgb, (imgsz, imgsz),
                                         interpolation=cv2.INTER_LINEAR)
                    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    if prec == "fp16":
                        img = img.half()
                    elif prec == "bf16":
                        img = img.to(torch.bfloat16)
                    img = img.to(device)

                pm = make_power_monitor(
                    args.enable_energy, args.sampler_exe,
                    csv_path=str(out_dir / f"power_{sw}x{sh}_{imgsz}_{prec}.csv"),
                    hz=args.ina_hz, hw=args.ina_hw,
                )
                meas = measure_loop(
                    iter_fn=preprocess_one,
                    n_warmup=args.n_warmup,
                    n_measure=args.n_measure,
                    power_monitor=pm,
                    sync_fn=sync_fn,
                )
                row = {
                    "src_width": sw,
                    "src_height": sh,
                    "in_pixels": sw * sh,
                    "dst_imgsz": imgsz,
                    "out_pixels": imgsz * imgsz,
                    "precision": prec,
                    "is_fp16": int(prec == "fp16"),
                    "resize": int(args.resize),
                    "device": device.type,
                }
                row.update(meas.as_row("preprocess"))
                csv.append(row)
                csv.flush()
                print(
                    f"  src={sw}x{sh} imgsz={imgsz} {prec}:  "
                    f"T={meas.latency_ms['mean']:7.4f} ms  "
                    f"E/iter={meas.energy_mj_per_iter:7.4f} mJ"
                )
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)

    print(f"\nDone. Wrote {len(csv)} rows → {csv.path}")


if __name__ == "__main__":
    main()
