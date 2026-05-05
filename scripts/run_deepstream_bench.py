"""
Single DeepStream benchmark run.

Usage (on Jetson):
    python3 scripts/run_deepstream_bench.py \\
        --imgsz 640 --precision fp32 --target-fps 0 --duration 60

    # Short smoke test (10 s, no energy):
    python3 scripts/run_deepstream_bench.py \\
        --imgsz 640 --precision fp32 --target-fps 0 --duration 10 \\
        --no-energy

Output: results/camera_bench/deepstream_<timestamp>/summary.json
        results/camera_bench/deepstream_<timestamp>/power_trace_raw.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepstream_bench.pipeline import run_deepstream_benchmark

DEFAULT_SAMPLER = str(PROJECT_ROOT / "src" / "energy_inference" / "tools" / "sample_ina3221")
DEFAULT_ENGINE_DIR = str(PROJECT_ROOT / "models" / "trt")
DEFAULT_CONFIG_DIR = str(PROJECT_ROOT / "configs" / "deepstream")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepStream benchmark — single run")
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--imgsz", type=int, default=640, choices=[320, 640])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--target-fps", type=int, default=0,
                   help="0 = unbounded, N = cap at N FPS")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Benchmark window in seconds (after warmup)")
    p.add_argument("--warmup", type=float, default=5.0,
                   help="Warmup duration in seconds")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--sampler-exe", default=DEFAULT_SAMPLER)
    p.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    p.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    p.add_argument("--out-dir", default=None,
                   help="Override output directory")
    p.add_argument("--no-energy", action="store_true",
                   help="Skip INA3221 energy measurement")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    engine_name = f"{args.model}_{args.imgsz}_{args.precision}.engine"
    engine_path = Path(args.engine_dir) / engine_name
    if not engine_path.exists():
        print(f"ERROR: TensorRT engine not found: {engine_path}", file=sys.stderr)
        print("Run:  bash scripts/setup_deepstream.sh", file=sys.stderr)
        sys.exit(1)

    config_name = f"nvinfer_{args.model}_{args.imgsz}_{args.precision}.txt"
    config_path = Path(args.config_dir) / config_name
    if not config_path.exists():
        print(f"ERROR: nvinfer config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_name = (
            f"deepstream_{args.model}_imgsz{args.imgsz}"
            f"_{args.precision}_fps{args.target_fps}_{ts}"
        )
        out_dir = PROJECT_ROOT / "results" / "camera_bench" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    power_csv = str(out_dir / "power_trace_raw.csv")
    sampler = None if args.no_energy else args.sampler_exe

    print("=" * 60)
    print("DeepStream Benchmark")
    print("=" * 60)
    print(f"  model     : {args.model}")
    print(f"  imgsz     : {args.imgsz}")
    print(f"  precision : {args.precision}")
    print(f"  target_fps: {args.target_fps} ({'unbounded' if args.target_fps == 0 else str(args.target_fps) + ' FPS'})")
    print(f"  duration  : {args.duration:.0f}s  warmup: {args.warmup:.0f}s")
    print(f"  engine    : {engine_path}")
    print(f"  nvinfer   : {config_path}")
    print(f"  output    : {out_dir}")
    print(f"  energy    : {'disabled' if args.no_energy else 'INA3221'}")
    print("=" * 60)

    t_wall_start = time.time()

    summary = run_deepstream_benchmark(
        device=args.device,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        nvinfer_config=str(config_path),
        warmup_s=args.warmup,
        duration_s=args.duration,
        sampler_exe=sampler,
        power_csv_path=power_csv,
    )

    t_wall_end = time.time()

    # Enrich summary with run metadata
    summary.update({
        "model": args.model,
        "imgsz": args.imgsz,
        "precision": args.precision,
        "target_fps": args.target_fps,
        "backend": "deepstream",
        "engine_path": str(engine_path),
        "nvinfer_config": str(config_path),
        "wall_time_s": round(t_wall_end - t_wall_start, 1),
    })

    # Write summary.json
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written → {summary_path}")

    # Print key results
    print("\n── Results ─────────────────────────────────────")
    print(f"  Frames      : {summary.get('n_frames', '?')}")
    print(f"  Actual FPS  : {summary.get('fps_mean', '?'):.2f}")
    print(f"  Duration    : {summary.get('actual_duration_s', '?'):.1f}s")
    if summary.get("energy_total_j") is not None:
        print(f"  Energy      : {summary['energy_total_j']:.2f} J")
        print(f"  Mean power  : {summary['mean_power_w']:.3f} W")
        print(f"  mJ/frame    : {summary['energy_per_frame_j']*1000:.2f}")
    print(f"  Status      : {summary.get('status', '?')}")
    if summary.get("error"):
        print(f"  Error       : {summary['error']}", file=sys.stderr)

    if summary.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
