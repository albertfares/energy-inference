"""
Grid sweep runner for the camera benchmark.

Run from the project root:
    PYTHONPATH=src python -m camera_bench.sweep [options]

Each configuration is run as a clean subprocess (no PyTorch state leakage between runs).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Default grid definitions
# ---------------------------------------------------------------------------

# Full default grid (§4.1)
DEFAULT_GRID: dict[str, list] = {
    "model": [
        "ssdlite320_mobilenet_v3_large",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "yolov8n",
    ],
    "width":  [640, 1280],
    "height": [480, 720],
    "precision": ["fp32", "fp16"],
    "target_fps": [0, 30],
    "output_stream": ["none", "rtp_h264_nvenc"],
}

# Streaming-comparison sub-grid: one representative config × all 4 streaming modes
STREAMING_COMPARISON_GRID: dict[str, list] = {
    "model":        ["yolov8n"],
    "width":        [640],
    "height":       [480],
    "precision":    ["fp16"],
    "target_fps":   [30],
    "output_stream": ["none", "mjpeg_cpu", "rtp_h264_sw", "rtp_h264_nvenc"],
}

GRIDS: dict[str, dict[str, list]] = {
    "default": DEFAULT_GRID,
    "streaming": STREAMING_COMPARISON_GRID,
    "minimal": {  # quick sanity-check (4 configs)
        "model":         ["ssdlite320_mobilenet_v3_large"],
        "width":         [640],
        "height":        [480],
        "precision":     ["fp32"],
        "target_fps":    [0],
        "output_stream": ["none"],
    },
}

# All pipeline stages that can appear across model backends.
_ALL_STAGES = [
    "capture", "preprocess", "infer", "infer_fused",
    "postprocess", "filter", "annotate", "encode",
]
_ALL_RAILS = ["cpu", "gpu", "io"]

SWEEP_SUMMARY_FIELDS = [
    # --- identity ---
    "run_idx", "repeat", "model", "width", "height", "precision",
    "target_fps", "output_stream",
    # --- run metadata ---
    "duration_s", "n_timed",
    # --- FPS ---
    "fps_mean", "fps_p50", "fps_p95", "fps_min", "fps_max",
    # --- total latency ---
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
    "latency_min_ms", "latency_max_ms",
    # --- per-stage latency (mean only — keeps CSV width sane) ---
    *[f"{s}_lat_mean_ms" for s in _ALL_STAGES],
    *[f"{s}_lat_p50_ms"  for s in _ALL_STAGES],
    *[f"{s}_lat_p95_ms"  for s in _ALL_STAGES],
    # --- top-level energy ---
    "energy_total_j", "mean_power_w",
    "energy_per_frame_j", "energy_per_inference_j",
    # --- per-rail energy ---
    *[f"{r}_rail_j" for r in _ALL_RAILS],
    # --- per-stage energy ---
    *[f"{s}_energy_j"   for s in _ALL_STAGES],
    *[f"{s}_energy_pct" for s in _ALL_STAGES],
    # --- idle ---
    "idle_j", "idle_pct",
    # --- sweep bookkeeping ---
    "status", "error", "run_dir",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m camera_bench.sweep",
        description="Run a grid of camera benchmark configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grid",
        default="default",
        help=(
            f"Grid name (built-in: {', '.join(GRIDS)}) "
            "or path to a JSON file with {axis: [values]} mapping."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=f"results/camera_bench/sweep_{datetime.now().strftime('%Y%m%d')}",
        help="Base directory for all sweep results.",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--cooldown-s", type=float, default=30.0)
    p.add_argument("--device", type=int, default=0, help="Camera device index.")
    p.add_argument("--fps", type=int, default=30, help="Camera requested FPS.")
    p.add_argument("--sampler-exe", default="src/energy_inference/tools/sample_ina3221")
    p.add_argument("--stream-host", default="127.0.0.1")
    p.add_argument("--stream-port", type=int, default=11111)
    p.add_argument("--stream-bitrate", type=int, default=2_000_000)
    p.add_argument(
        "--no-energy", dest="enable_energy", action="store_false", default=True
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full config grid without running anything.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs whose run_dir already exists.",
    )
    return p


# ---------------------------------------------------------------------------
# Grid enumeration
# ---------------------------------------------------------------------------

def load_grid(grid_arg: str) -> dict[str, list]:
    if grid_arg in GRIDS:
        return GRIDS[grid_arg]
    p = Path(grid_arg)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    raise ValueError(
        f"Unknown grid {grid_arg!r}. "
        f"Built-in options: {list(GRIDS)}. Or pass a JSON file path."
    )


def enumerate_configs(grid: dict[str, list]) -> list[dict]:
    """Return all (model, res, prec, fps, stream) combos as dicts."""
    axes = list(grid.keys())
    values = [grid[k] for k in axes]
    configs = []
    for combo in product(*values):
        cfg = dict(zip(axes, combo))
        # Enforce: width/height must go together sensibly
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

def _build_cli_cmd(
    cfg: dict,
    run_name: str,
    out_dir: str,
    base_args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "camera_bench.cli",
        "--bench-mode",
        "--model", cfg["model"],
        "--width", str(cfg.get("width", 640)),
        "--height", str(cfg.get("height", 480)),
        "--precision", cfg.get("precision", "fp32"),
        "--target-fps", str(cfg.get("target_fps", 0)),
        "--output-stream", cfg.get("output_stream", "none"),
        "--duration-s", str(base_args.duration_s),
        "--warmup-frames", str(base_args.warmup_frames),
        "--fps", str(base_args.fps),
        "--device", str(base_args.device),
        "--sampler-exe", base_args.sampler_exe,
        "--stream-host", base_args.stream_host,
        "--stream-port", str(base_args.stream_port),
        "--stream-bitrate", str(base_args.stream_bitrate),
        "--out-dir", out_dir,
        "--run-name", run_name,
    ]
    if not base_args.enable_energy:
        cmd.append("--no-energy")
    return cmd


def _run_single(
    cmd: list[str],
    run_dir: Path,
    timeout_s: float,
) -> tuple[str, str]:
    """
    Run a single benchmark subprocess.

    Returns (status, error_message).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    log_path = run_dir / "log.txt"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w") as log_f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        if result.returncode != 0:
            return "failed", f"exit code {result.returncode}"
        return "ok", ""
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded {timeout_s:.0f}s"
    except Exception as exc:
        return "error", str(exc)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _parse_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open() as f:
        return json.load(f)


def _extract_sweep_row(
    run_idx: int,
    repeat: int,
    cfg: dict,
    status: str,
    error: str,
    run_dir: Path,
    summary: dict,
) -> dict:
    fps = summary.get("fps", {})
    lat_total = summary.get("latency_ms", {}).get("total", {})
    lat_stage = summary.get("latency_ms", {}).get("per_stage", {})
    en = summary.get("energy", {})
    per_stage_j   = en.get("per_stage_j", {})
    per_stage_pct = en.get("per_stage_pct", {})
    per_rail_j    = en.get("per_rail_j", {})

    row: dict = {
        # identity
        "run_idx":       run_idx,
        "repeat":        repeat,
        "model":         cfg.get("model", ""),
        "width":         cfg.get("width", ""),
        "height":        cfg.get("height", ""),
        "precision":     cfg.get("precision", ""),
        "target_fps":    cfg.get("target_fps", ""),
        "output_stream": cfg.get("output_stream", ""),
        # run metadata
        "duration_s": summary.get("duration_s", ""),
        "n_timed":    summary.get("n_timed", ""),
        # FPS
        "fps_mean": fps.get("mean", ""),
        "fps_p50":  fps.get("p50", ""),
        "fps_p95":  fps.get("p95", ""),
        "fps_min":  fps.get("min", ""),
        "fps_max":  fps.get("max", ""),
        # total latency
        "latency_mean_ms": lat_total.get("mean", ""),
        "latency_p50_ms":  lat_total.get("p50",  ""),
        "latency_p95_ms":  lat_total.get("p95",  ""),
        "latency_min_ms":  lat_total.get("min",  ""),
        "latency_max_ms":  lat_total.get("max",  ""),
        # top-level energy
        "energy_total_j":        en.get("total_j", ""),
        "mean_power_w":          en.get("mean_power_w", ""),
        "energy_per_frame_j":    en.get("energy_per_frame_j", ""),
        "energy_per_inference_j": en.get("energy_per_inference_j", ""),
        # idle
        "idle_j":   en.get("idle_j", ""),
        "idle_pct": en.get("idle_pct", ""),
        # sweep bookkeeping
        "status":  status,
        "error":   error,
        "run_dir": str(run_dir),
    }

    # per-stage latency
    for s in _ALL_STAGES:
        s_lat = lat_stage.get(s, {})
        row[f"{s}_lat_mean_ms"] = s_lat.get("mean", "")
        row[f"{s}_lat_p50_ms"]  = s_lat.get("p50",  "")
        row[f"{s}_lat_p95_ms"]  = s_lat.get("p95",  "")

    # per-rail energy
    for r in _ALL_RAILS:
        row[f"{r}_rail_j"] = per_rail_j.get(r, "")

    # per-stage energy
    for s in _ALL_STAGES:
        row[f"{s}_energy_j"]   = per_stage_j.get(s, "")
        row[f"{s}_energy_pct"] = per_stage_pct.get(s, "")

    return row


def _append_sweep_csv(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SWEEP_SUMMARY_FIELDS, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    grid = load_grid(args.grid)
    configs = enumerate_configs(grid)
    n_configs = len(configs)
    n_total = n_configs * args.repeats

    print(f"Sweep: {args.grid!r} grid  |  {n_configs} configs × {args.repeats} repeats = {n_total} runs")
    print(f"Output: {args.out_dir}")

    if args.dry_run:
        print("\n--- DRY RUN: config list ---")
        for i, cfg in enumerate(configs):
            for r in range(args.repeats):
                print(f"  [{i * args.repeats + r + 1}/{n_total}] {cfg}  repeat={r}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "sweep_summary.csv"
    failures_csv = out_dir / "sweep_failures.csv"

    # Estimate total time
    run_s = args.duration_s + 30  # approx: bench + cleanup
    cooldown_s = args.cooldown_s
    est_h = n_total * (run_s + cooldown_s) / 3600
    print(f"Estimated time: ~{est_h:.1f}h  (at {run_s:.0f}s/run + {cooldown_s:.0f}s cooldown)")

    global_run_idx = 0
    t_sweep_start = time.time()

    for cfg_idx, cfg in enumerate(configs):
        for repeat in range(args.repeats):
            global_run_idx += 1
            elapsed_h = (time.time() - t_sweep_start) / 3600
            remain = n_total - global_run_idx
            eta_h = remain * (run_s + cooldown_s) / 3600

            # Build run name — deterministic so --resume can find it by path.
            # Timestamp is NOT part of the directory name; it lives in summary.json.
            run_name = (
                f"{cfg.get('model','?')}_{cfg.get('width','?')}x{cfg.get('height','?')}"
                f"_{cfg.get('precision','?')}_{cfg.get('output_stream','?')}"
                f"_fps{cfg.get('target_fps',0)}_r{repeat}"
            )
            run_dir = out_dir / run_name

            print(
                f"\n[{global_run_idx}/{n_total}] elapsed={elapsed_h:.2f}h  "
                f"eta={eta_h:.2f}h  cfg={cfg}  repeat={repeat}"
            )

            if args.resume and run_dir.exists() and (run_dir / "summary.json").exists():
                print(f"  Skipping (resume): {run_dir}")
                continue

            cmd = _build_cli_cmd(cfg, run_name, str(out_dir), args)
            timeout_s = args.duration_s * 3 + 120  # generous timeout

            status, error = _run_single(cmd, run_dir, timeout_s)
            summary = _parse_summary(run_dir)

            sweep_row = _extract_sweep_row(
                global_run_idx, repeat, cfg, status, error, run_dir, summary
            )
            _append_sweep_csv(sweep_csv, sweep_row)

            if status != "ok":
                print(f"  FAILED: {error}")
                _append_sweep_csv(failures_csv, sweep_row)
            else:
                fps = summary.get("fps", {})
                en = summary.get("energy", {})
                fps_mean = fps.get("mean")
                total_j = en.get("total_j")
                if isinstance(fps_mean, (int, float)) and isinstance(total_j, (int, float)):
                    print(f"  OK  fps_mean={float(fps_mean):.1f}  energy={float(total_j):.2f}J")
                else:
                    print("  OK")

            # Cooldown (skip after last run)
            if global_run_idx < n_total:
                print(f"  Cooldown {args.cooldown_s:.0f}s ...", flush=True)
                time.sleep(args.cooldown_s)

    print(f"\nSweep complete. Summary: {sweep_csv}")

    # Write sweep_summary.json aggregating all runs
    _write_sweep_summary_json(sweep_csv, out_dir / "sweep_summary.json")


def _write_sweep_summary_json(csv_path: Path, json_path: Path) -> None:
    if not csv_path.exists():
        return
    rows: list[dict] = []
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    with json_path.open("w") as f:
        json.dump({"n_runs": len(rows), "runs": rows}, f, indent=2)


if __name__ == "__main__":
    main()
