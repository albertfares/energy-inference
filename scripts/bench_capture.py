"""
Benchmark A — isolated CAPTURE stage (pure camera, no model).

Why this exists
---------------
The capture sub-predictor must be a lego brick that depends ONLY on the camera
hardware and the requested frame rate — never on which detector is downstream.
This script runs a bare ``cap.read()`` loop with no preprocessing, no model and
no postprocessing, and measures latency + board energy via INA3221.

What it recovers
----------------
For each (resolution, target_fps):

  * target_fps == 0  (unbounded): ``cap.read()`` blocks until the sensor
    delivers the next frame, so the per-iteration latency ≈ the camera frame
    PERIOD. This gives ``T_camera_period_ms`` for that resolution.

  * target_fps  > 0  (throttled): between reads we sleep ~ (1/fps), during which
    the driver buffers a frame. The next read then returns almost immediately,
    so the latency ≈ pure ``T_decode_ms`` (YUV→BGR copy out of the buffer).
    Because the read is a tiny fraction of the period, the mean board power over
    the window ≈ the platform IDLE / SLEEP power — the term the combination
    layer charges for inter-frame waits.

Output
------
results/isolated_bench/capture/<run>/capture_bench.csv
    one row per (width,height,target_fps) with latency stats, per-iter energy,
    and mean_power_w (used as the sleep-power estimate at high throttle).

Run on the Jetson, from the project root::

    PYTHONPATH=src python scripts/bench_capture.py \
        --device 0 --resolutions 640x480 1280x720 \
        --target-fps 0 5 10 15 30 --n-measure 600
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

from camera_bench.capture import open_camera, validate_camera_resolution
from camera_bench.isolated import IsolatedCSV, make_power_monitor, measure_loop


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated capture-stage benchmark.")
    p.add_argument("--device", type=int, default=0, help="/dev/videoN index.")
    p.add_argument(
        "--resolutions", nargs="+", default=["640x480"],
        help="Capture resolutions as WxH (space separated).",
    )
    p.add_argument(
        "--target-fps", nargs="+", type=int, default=[0, 5, 10, 15, 30],
        help="FPS caps to sweep (0 = unbounded / camera-limited).",
    )
    p.add_argument("--n-warmup", type=int, default=60)
    p.add_argument("--n-measure", type=int, default=600)
    p.add_argument("--cooldown-s", type=float, default=5.0)
    # energy
    p.add_argument("--no-energy", dest="enable_energy", action="store_false")
    p.add_argument("--ina-hz", type=int, default=1000)
    p.add_argument("--ina-hw", default="all", choices=["cpu", "gpu", "io", "both", "all"])
    p.add_argument(
        "--sampler-exe",
        default="src/energy_inference/tools/sample_ina3221",
    )
    p.add_argument("--out-dir", default="results/isolated_bench/capture")
    p.add_argument("--run-name", default="auto")
    return p


def main() -> None:
    import time
    args = build_parser().parse_args()

    run_name = args.run_name
    if run_name == "auto":
        run_name = "capture_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = IsolatedCSV(out_dir / "capture_bench.csv")
    print(f"Results: {out_dir}")

    resolutions = [parse_resolution(r) for r in args.resolutions]

    for (req_w, req_h) in resolutions:
        # Open once per resolution; sweep FPS caps without re-opening so the
        # only thing that changes between rows is the pacing.
        cap, act_w, act_h, act_fps = open_camera(args.device, req_w, req_h, 30)
        validate_camera_resolution(args.device, act_w, act_h, req_w, req_h)
        print(f"\nCamera {act_w}x{act_h} @ driver {act_fps:.1f} fps")

        def read_one() -> None:
            ok, _frame = cap.read()
            if not ok:
                raise RuntimeError("cap.read() failed — camera disconnected?")

        for tfps in args.target_fps:
            period_s = (1.0 / tfps) if tfps > 0 else 0.0
            pm = make_power_monitor(
                args.enable_energy, args.sampler_exe,
                csv_path=str(out_dir / f"power_{act_w}x{act_h}_fps{tfps}.csv"),
                hz=args.ina_hz, hw=args.ina_hw,
            )
            meas = measure_loop(
                iter_fn=read_one,
                n_warmup=args.n_warmup,
                n_measure=args.n_measure,
                power_monitor=pm,
                sync_fn=None,
                target_period_s=period_s,
            )
            row = {
                "width": act_w,
                "height": act_h,
                "pixels": act_w * act_h,
                "target_fps": tfps,
                "regime": "camera_limited" if tfps == 0 else "throttled",
                "driver_fps": round(act_fps, 3),
            }
            row.update(meas.as_row("capture"))
            csv.append(row)
            csv.flush()
            print(
                f"  fps_cap={tfps:>3}  "
                f"T_capture={meas.latency_ms['mean']:7.3f} ms  "
                f"E/iter={meas.energy_mj_per_iter:7.3f} mJ  "
                f"P_mean={meas.mean_power_w:6.3f} W"
            )
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

        cap.release()

    print(f"\nDone. Wrote {len(csv)} rows → {csv.path}")
    print("Interpretation:")
    print("  • fps_cap=0 rows   → T_capture ≈ camera frame PERIOD (per resolution)")
    print("  • throttled rows   → T_capture ≈ T_decode; P_mean ≈ idle/sleep power")


if __name__ == "__main__":
    main()
