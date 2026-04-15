"""
Single-config camera benchmark entry point.

Run from the project root:
    PYTHONPATH=src python -m camera_bench.cli --bench-mode [options]

Legacy live-detect (no benchmarking): use scripts/live_detect_ssdlite.py directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .models import SUPPORTED_MODELS
from .results import make_run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m camera_bench.cli",
        description="Camera-pipeline benchmark for Jetson edge inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Benchmark mode switch ----
    p.add_argument(
        "--bench-mode",
        action="store_true",
        help="Enable benchmark mode (required for timed runs).",
    )

    # ---- Model / inference ----
    p.add_argument(
        "--model",
        default="ssdlite320_mobilenet_v3_large",
        choices=SUPPORTED_MODELS,
    )
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available.")
    p.add_argument("--score-threshold", type=float, default=0.45)
    p.add_argument("--iou-threshold", type=float, default=0.45)
    p.add_argument("--max-detections", type=int, default=8)
    p.add_argument(
        "--yolo-imgsz", type=int, default=640, help="YOLO inference image size."
    )

    # ---- Camera ----
    p.add_argument("--device", type=int, default=0, help="Video device index (/dev/videoN).")
    p.add_argument("--width", type=int, default=640, help="Requested capture width.")
    p.add_argument("--height", type=int, default=480, help="Requested capture height.")
    p.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")

    # ---- Benchmark control ----
    p.add_argument("--duration-s", type=float, default=120.0, help="Timed run length (s).")
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument(
        "--target-fps",
        type=int,
        default=0,
        help="Pace inference to this FPS (0 = unbounded).",
    )
    p.add_argument("--repeat", type=int, default=1, help="Repeat this config N times.")
    p.add_argument("--cooldown-s", type=float, default=30.0, help="Idle between repeats.")

    # ---- Output streaming ----
    p.add_argument(
        "--output-stream",
        default="none",
        choices=["none", "mjpeg_cpu", "rtp_h264_sw", "rtp_h264_nvenc"],
        help="Output streaming mode.",
    )
    p.add_argument("--stream-host", default="127.0.0.1")
    p.add_argument("--stream-port", type=int, default=11111)
    p.add_argument("--stream-bitrate", type=int, default=2_000_000, help="H.264 bitrate (bps).")
    p.add_argument("--rtp-sdp", default="cam_bench.sdp")
    p.add_argument("--mjpeg-host", default="127.0.0.1")
    p.add_argument("--mjpeg-port", type=int, default=8080)

    # ---- Legacy pass-through (non-bench mode) ----
    p.add_argument("--show", action="store_true", help="Show GUI preview (non-bench only).")
    p.add_argument("--serve-mjpeg", action="store_true", help="Legacy MJPEG flag.")
    p.add_argument("--stream-rtp", action="store_true", help="Legacy RTP flag.")
    p.add_argument("--rtp-host", default=None)
    p.add_argument("--rtp-port", type=int, default=11111)

    # ---- Energy / INA3221 ----
    p.add_argument(
        "--stage-energy",
        action="store_true",
        default=True,
        help="Enable per-stage energy attribution.",
    )
    p.add_argument(
        "--no-stage-energy",
        dest="stage_energy",
        action="store_false",
        help="Disable per-stage energy attribution.",
    )
    p.add_argument(
        "--enable-energy",
        action="store_true",
        default=True,
        help="Enable INA3221 power sampling.",
    )
    p.add_argument(
        "--no-energy",
        dest="enable_energy",
        action="store_false",
        help="Disable INA3221 power sampling.",
    )
    p.add_argument("--ina-hz", type=int, default=1000)
    p.add_argument("--ina-hw", default="all", choices=["cpu", "gpu", "io", "both", "all"])
    p.add_argument(
        "--sampler-exe",
        default="src/energy_inference/tools/sample_ina3221",
        help="Path to INA3221 sampler executable.",
    )
    p.add_argument("--power-csv", default=None, help="Override power trace CSV path.")

    # ---- Results ----
    p.add_argument(
        "--out-dir",
        default="results/camera_bench",
        help="Base directory for results.",
    )
    p.add_argument(
        "--run-name",
        default="auto",
        help="Run directory name (default: auto-generated).",
    )

    # ---- Sustained mode ----
    p.add_argument(
        "--sustained",
        action="store_true",
        help="Run in sustained mode (20 min, 1 s power+fps log).",
    )
    p.add_argument(
        "--sustained-duration-s",
        type=float,
        default=1200.0,
        help="Duration for sustained mode.",
    )

    return p


def args_to_cfg(args: argparse.Namespace) -> dict:
    """Convert parsed args into a plain config dict for pipeline.run_benchmark()."""
    return {
        "model": args.model,
        "precision": args.precision,
        "cpu": args.cpu,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "max_detections": args.max_detections,
        "yolo_imgsz": args.yolo_imgsz,
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration_s": args.duration_s,
        "warmup_frames": args.warmup_frames,
        "target_fps": args.target_fps,
        "output_stream": args.output_stream,
        "stream_host": args.stream_host,
        "stream_port": args.stream_port,
        "stream_bitrate": args.stream_bitrate,
        "rtp_sdp": args.rtp_sdp,
        "mjpeg_host": args.mjpeg_host,
        "mjpeg_port": args.mjpeg_port,
        "stage_energy": args.stage_energy,
        "enable_energy": args.enable_energy,
        "ina_hz": args.ina_hz,
        "ina_hw": args.ina_hw,
        "sampler_exe": args.sampler_exe,
        "power_csv": args.power_csv,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.bench_mode:
        # Delegate to legacy live-detect script for non-bench use
        print(
            "No --bench-mode flag. For live detection use scripts/live_detect_ssdlite.py.\n"
            "Pass --bench-mode to enable benchmarking.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    # bench mode: --show is forced off
    if args.show:
        print(
            "[bench-mode] --show ignored; GUI preview is not part of the "
            "deployment scenario being measured.",
            file=sys.stderr,
        )

    if args.serve_mjpeg or args.stream_rtp:
        print(
            "[bench-mode] --serve-mjpeg / --stream-rtp are ignored in bench mode; "
            "use --output-stream instead.",
            file=sys.stderr,
        )

    cfg = args_to_cfg(args)

    import time
    from .pipeline import run_benchmark
    from .results import ResultsDir

    for repeat_idx in range(args.repeat):
        if args.repeat > 1:
            print(f"\n{'='*62}\nRepeat {repeat_idx + 1}/{args.repeat}\n{'='*62}")

        # Resolve run name
        run_name = args.run_name
        if run_name == "auto":
            run_name = make_run_name(
                model=args.model,
                width=args.width,
                height=args.height,
                precision=args.precision,
                stream_mode=args.output_stream,
                target_fps=args.target_fps,
            )
            if args.repeat > 1:
                run_name += f"_r{repeat_idx}"

        results_dir = ResultsDir(args.out_dir, run_name)
        print(f"Results: {results_dir.run_dir}")

        try:
            run_benchmark(cfg, results_dir)
        except Exception as exc:
            print(f"ERROR: benchmark failed: {exc}", file=sys.stderr)
            raise

        if repeat_idx < args.repeat - 1:
            print(f"Cooldown: {args.cooldown_s:.0f}s ...", flush=True)
            time.sleep(args.cooldown_s)


if __name__ == "__main__":
    main()
