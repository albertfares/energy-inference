"""
DeepStream FPS sweep — runs all (imgsz × precision × target_fps × repeats)
combinations and writes a sweep_summary.csv comparable to the PyTorch one.

Usage (on Jetson — run AFTER setup_deepstream.sh):
    python3 scripts/run_deepstream_sweep.py \\
        --grid grids/deepstream_fps_sweep.json

    # Resume a partial sweep (skips existing run dirs):
    python3 scripts/run_deepstream_sweep.py \\
        --grid grids/deepstream_fps_sweep.json --resume

Output:
    results/camera_bench/deepstream_yolov8n_fps_sweep_MAXN_<timestamp>/
        sweep_summary.csv       ← one row per run, directly comparable to
                                   the PyTorch sweep_summary.csv
        <run_name>/
            summary.json
            power_trace_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── CSV schema (subset of PyTorch sweep_summary — directly comparable) ────────
SUMMARY_FIELDS = [
    "run_idx", "repeat", "model", "backend",
    "width", "height", "yolo_imgsz", "precision",
    "target_fps", "output_stream",
    "duration_s", "n_frames",
    "fps_mean", "fps_p50", "fps_p95", "fps_min", "fps_max",
    "energy_total_j", "mean_power_w", "energy_per_frame_j",
    "cpu_rail_j", "gpu_rail_j", "io_rail_j",
    "status", "error", "run_dir",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepStream FPS sweep")
    p.add_argument("--grid", required=True,
                   help="Path to grid JSON (e.g. grids/deepstream_fps_sweep.json)")
    p.add_argument("--resume", action="store_true",
                   help="Skip run dirs that already contain a summary.json")
    p.add_argument("--out-base", default=None,
                   help="Override base output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned runs without executing")
    return p.parse_args()


def load_grid(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def expand_grid(grid: dict) -> list[dict]:
    """Return one dict per (imgsz, precision, target_fps, repeat) combination."""
    imgsz_list = grid.get("imgsz", [640])
    if isinstance(imgsz_list, int):
        imgsz_list = [imgsz_list]

    precision_list = grid.get("precision", ["fp32"])
    if isinstance(precision_list, str):
        precision_list = [precision_list]

    fps_list = grid.get("target_fps", [0, 5, 10, 15, 20, 25, 30])
    repeats = grid.get("repeats", 3)

    runs = []
    for imgsz, prec, fps, repeat in product(
        imgsz_list, precision_list, fps_list, range(repeats)
    ):
        runs.append({
            "model":       grid.get("model", "yolov8n"),
            "device":      grid.get("device", "/dev/video0"),
            "width":       grid.get("width", 640),
            "height":      grid.get("height", 480),
            "imgsz":       imgsz,
            "precision":   prec,
            "target_fps":  fps,
            "repeat":      repeat,
            "duration_s":  grid.get("duration_s", 60),
            "warmup_s":    grid.get("warmup_s", 5),
            "sampler_exe": grid.get("sampler_exe",
                                    "src/energy_inference/tools/sample_ina3221"),
        })
    return runs


def run_name_for(cfg: dict) -> str:
    """Deterministic run directory name (no timestamp → --resume works)."""
    return (
        f"deepstream_{cfg['model']}"
        f"_{cfg['width']}x{cfg['height']}"
        f"_imgsz{cfg['imgsz']}"
        f"_{cfg['precision']}"
        f"_none"                       # output_stream=none (always for DS bench)
        f"_fps{cfg['target_fps']}"
        f"_r{cfg['repeat']}"
    )


def run_single(cfg: dict, run_dir: Path) -> dict:
    """Invoke run_deepstream_bench.py as a subprocess. Returns parsed summary dict."""
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_deepstream_bench.py"),
        "--model",      cfg["model"],
        "--imgsz",      str(cfg["imgsz"]),
        "--precision",  cfg["precision"],
        "--target-fps", str(cfg["target_fps"]),
        "--duration",   str(cfg["duration_s"]),
        "--warmup",     str(cfg["warmup_s"]),
        "--device",     cfg["device"],
        "--width",      str(cfg["width"]),
        "--height",     str(cfg["height"]),
        "--sampler-exe", str(PROJECT_ROOT / cfg["sampler_exe"]),
        "--out-dir",    str(run_dir),
    ]

    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s  exit_code={result.returncode}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    return {
        "status": "failed",
        "error": f"exit code {result.returncode}, no summary.json",
    }


def summary_to_row(run_idx: int, cfg: dict, s: dict, run_dir: Path) -> dict:
    """Flatten a summary dict into a CSV row."""
    return {
        "run_idx":        run_idx,
        "repeat":         cfg["repeat"],
        "model":          cfg["model"],
        "backend":        "deepstream",
        "width":          cfg["width"],
        "height":         cfg["height"],
        "yolo_imgsz":     cfg["imgsz"],
        "precision":      cfg["precision"],
        "target_fps":     cfg["target_fps"],
        "output_stream":  "none",
        "duration_s":     s.get("actual_duration_s", ""),
        "n_frames":       s.get("n_frames", ""),
        "fps_mean":       s.get("fps_mean", ""),
        "fps_p50":        s.get("fps_p50", ""),
        "fps_p95":        s.get("fps_p95", ""),
        "fps_min":        s.get("fps_min", ""),
        "fps_max":        s.get("fps_max", ""),
        "energy_total_j": s.get("energy_total_j", ""),
        "mean_power_w":   s.get("mean_power_w", ""),
        "energy_per_frame_j": s.get("energy_per_frame_j", ""),
        "cpu_rail_j":     s.get("cpu_rail_j", ""),
        "gpu_rail_j":     s.get("gpu_rail_j", ""),
        "io_rail_j":      s.get("io_rail_j", ""),
        "status":         s.get("status", "failed"),
        "error":          s.get("error", ""),
        "run_dir":        str(run_dir.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    args = parse_args()
    grid = load_grid(args.grid)
    runs = expand_grid(grid)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"deepstream_{grid.get('model', 'yolov8n')}_fps_sweep_MAXN_{ts}"

    if args.out_base:
        sweep_dir = Path(args.out_base)
    else:
        sweep_dir = PROJECT_ROOT / "results" / "camera_bench" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = sweep_dir / "sweep_summary.csv"
    print(f"\n{'='*60}")
    print(f"DeepStream FPS sweep — {len(runs)} runs")
    print(f"Output: {sweep_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, cfg in enumerate(runs, 1):
            print(f"  [{i:3d}/{len(runs)}] {run_name_for(cfg)}")
        return

    n_ok = n_fail = n_skip = 0

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for run_idx, cfg in enumerate(runs, 1):
            rname = run_name_for(cfg)
            run_dir = sweep_dir / rname

            print(f"\n[{run_idx:3d}/{len(runs)}] {rname}")

            # Resume: skip if summary.json already exists and is valid
            existing = run_dir / "summary.json"
            if args.resume and existing.exists():
                try:
                    with open(existing) as ef:
                        s = json.load(ef)
                    if s.get("status") == "ok":
                        row = summary_to_row(run_idx, cfg, s, run_dir)
                        writer.writerow(row)
                        f.flush()
                        n_skip += 1
                        print(f"  SKIP (existing ok run)")
                        continue
                except Exception:
                    pass

            s = run_single(cfg, run_dir)
            row = summary_to_row(run_idx, cfg, s, run_dir)
            writer.writerow(row)
            f.flush()

            if s.get("status") == "ok":
                n_ok += 1
                fps = s.get("fps_mean", "?")
                epf = s.get("energy_per_frame_j")
                epf_str = f"{epf*1000:.0f}mJ" if epf else "n/a"
                pw = s.get("mean_power_w")
                pw_str = f"{pw:.2f}W" if pw else "n/a"
                print(f"  OK — fps={fps:.2f}  energy/frame={epf_str}  power={pw_str}")
            else:
                n_fail += 1
                print(f"  FAILED — {s.get('error', 'unknown')}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Sweep complete: {n_ok} ok  {n_fail} failed  {n_skip} skipped")
    print(f"Summary CSV: {summary_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
