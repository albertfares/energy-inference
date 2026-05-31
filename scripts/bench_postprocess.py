"""
Benchmark D — isolated POSTPROCESS / NMS stage (synthetic boxes, no model).

Why this exists
---------------
The supervisor asked specifically for the postprocess sub-predictor to be driven
by the NUMBER OF OBJECTS, because non-maximum suppression cost grows with the
number of candidate detections (≈ O(N²) in the pairwise IoU step). The only way
to make that a clean lego brick is to feed NMS synthetic boxes with a controlled
count and measure latency + energy directly — no model, no camera.

What one iteration does (mirrors the real postprocess path)
-----------------------------------------------------------
  1. score-threshold filter      scores >= score_threshold
  2. class-aware NMS              torchvision.ops.batched_nms(boxes, scores, labels, iou)
  3. top-k cap                    keep[:max_detections]
  4. device→host copy            .cpu()   (the D2H the pipeline pays for)

Sweep axes
----------
  * n_boxes        number of candidate detections fed to NMS (the headline
                   "number of objects" feature)
  * iou_threshold  NMS IoU
  * score_threshold
  * max_detections top-k cap

Output
------
results/isolated_bench/postprocess/<run>/postprocess_bench.csv
    one row per (n_boxes, iou_threshold, score_threshold, max_detections), with
    n_kept (objects surviving NMS) recorded alongside.

Run on the Jetson::

    PYTHONPATH=src python scripts/bench_postprocess.py \
        --n-boxes 5 20 50 100 300 1000 3000 \
        --iou-threshold 0.45 --n-measure 2000
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated postprocess/NMS-stage benchmark.")
    p.add_argument("--n-boxes", nargs="+", type=int,
                   default=[5, 20, 50, 100, 300, 1000, 3000])
    p.add_argument("--iou-threshold", nargs="+", type=float, default=[0.3, 0.45, 0.6])
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--max-detections", type=int, default=300)
    p.add_argument("--n-classes", type=int, default=80)
    p.add_argument("--canvas", type=int, default=640,
                   help="Coordinate range for synthetic boxes (px).")
    p.add_argument("--cpu", action="store_true",
                   help="Run NMS on CPU (default: CUDA if available).")
    p.add_argument("--n-warmup", type=int, default=100)
    p.add_argument("--n-measure", type=int, default=2000)
    p.add_argument("--cooldown-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-energy", dest="enable_energy", action="store_false")
    p.add_argument("--ina-hz", type=int, default=1000)
    p.add_argument("--ina-hw", default="all", choices=["cpu", "gpu", "io", "both", "all"])
    p.add_argument("--sampler-exe", default="src/energy_inference/tools/sample_ina3221")
    p.add_argument("--out-dir", default="results/isolated_bench/postprocess")
    p.add_argument("--run-name", default="auto")
    return p


def make_synthetic_boxes(n: int, n_classes: int, canvas: int, device, seed: int):
    """Random valid xyxy boxes, scores in [0,1], class labels."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * canvas
    wh = torch.rand(n, 2, generator=g) * (canvas * 0.25) + 1.0
    boxes = torch.cat([xy, xy + wh], dim=1).to(device)
    scores = torch.rand(n, generator=g).to(device)
    labels = torch.randint(0, max(1, n_classes), (n,), generator=g).to(device)
    return boxes, scores, labels


def main() -> None:
    import time
    import torch
    from torchvision.ops import batched_nms

    args = build_parser().parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    sync_fn = torch_cuda_sync_fn(device)
    print(f"Torch device: {device}")

    run_name = args.run_name
    if run_name == "auto":
        run_name = "postprocess_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = IsolatedCSV(out_dir / "postprocess_bench.csv")
    print(f"Results: {out_dir}")

    for iou in args.iou_threshold:
        for n in args.n_boxes:
            boxes, scores, labels = make_synthetic_boxes(
                n, args.n_classes, args.canvas, device, args.seed)
            score_mask = scores >= args.score_threshold
            n_filtered = int(score_mask.sum().item())  # boxes actually fed to NMS

            # Measure how many survive once (for the n_kept column).
            with torch.no_grad():
                b0, s0, l0 = boxes[score_mask], scores[score_mask], labels[score_mask]
                keep0 = batched_nms(b0, s0, l0, iou)[: args.max_detections]
            n_kept = int(keep0.numel())

            def postprocess_one() -> None:
                with torch.no_grad():
                    b = boxes[score_mask]
                    s = scores[score_mask]
                    l = labels[score_mask]
                    keep = batched_nms(b, s, l, iou)[: args.max_detections]
                    _ = b[keep].cpu()
                    _ = s[keep].cpu()
                    _ = l[keep].cpu()

            pm = make_power_monitor(
                args.enable_energy, args.sampler_exe,
                csv_path=str(out_dir / f"power_n{n}_iou{iou}.csv"),
                hz=args.ina_hz, hw=args.ina_hw,
            )
            meas = measure_loop(
                iter_fn=postprocess_one,
                n_warmup=args.n_warmup,
                n_measure=args.n_measure,
                power_monitor=pm,
                sync_fn=sync_fn,
            )
            row = {
                "n_boxes": n,
                "n_filtered": n_filtered,
                "n_kept": n_kept,
                "iou_threshold": iou,
                "score_threshold": args.score_threshold,
                "max_detections": args.max_detections,
                "n_classes": args.n_classes,
                "device": device.type,
            }
            row.update(meas.as_row("postprocess"))
            csv.append(row)
            csv.flush()
            print(
                f"  n_boxes={n:>5} (kept={n_kept:>3}) iou={iou}:  "
                f"T={meas.latency_ms['mean']:8.4f} ms  "
                f"E/iter={meas.energy_mj_per_iter:8.4f} mJ"
            )
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

    print(f"\nDone. Wrote {len(csv)} rows → {csv.path}")
    print("Feature note: 'n_boxes' is the number-of-objects driver the NMS "
          "predictor regresses on; 'n_kept' is recorded for reference.")


if __name__ == "__main__":
    main()
