"""
Camera benchmark plotting script.

Usage:
    python scripts/plot_camera_bench.py --sweep-dir results/camera_bench/sweep_20240415
    python scripts/plot_camera_bench.py --sweep-dir ... --out-dir results/plots/camera_bench

Produces:
    1. fps_vs_target.png        — FPS achieved vs target FPS per model
    2. energy_per_frame.png     — Energy/frame (mJ) bar chart by model × precision
    3. stage_energy_breakdown.png — Stage energy stacked bar per model
    4. streaming_overhead.png   — Streaming mode comparison (energy + per-stage)
    5. sustained_timeline.png   — Power + FPS over time for sustained runs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sweep_csv(sweep_dir: Path) -> list[dict]:
    p = sweep_dir / "sweep_summary.csv"
    if not p.exists():
        print(f"sweep_summary.csv not found in {sweep_dir}", file=sys.stderr)
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def _load_summary(run_dir: str | Path) -> dict:
    p = Path(run_dir) / "summary.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


def _load_stage_energy(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "stage_energy.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _savefig(fig: plt.Figure, path: Path, title: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 1: FPS achieved vs target FPS
# ---------------------------------------------------------------------------

def plot_fps_vs_target(rows: list[dict], out_dir: Path) -> None:
    """Line chart: achieved mean FPS vs target FPS, one line per model."""
    models = sorted({r["model"] for r in rows})
    target_fps_vals = sorted({int(_safe_float(r["target_fps"])) for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        xs, ys = [], []
        for tfps in target_fps_vals:
            subset = [
                r for r in rows
                if r["model"] == model
                and int(_safe_float(r["target_fps"])) == tfps
                and r["status"] == "ok"
            ]
            if not subset:
                continue
            fps_vals = [_safe_float(r["fps_mean"]) for r in subset]
            xs.append(tfps if tfps > 0 else max(fps_vals) * 1.05)
            ys.append(np.mean(fps_vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=model.replace("_", "\n"), color=color)

    # Ideal line
    xlim = ax.get_xlim()
    ax.plot([0, 60], [0, 60], "k--", alpha=0.3, label="ideal")
    ax.set_xlabel("Target FPS (0 = unbounded)")
    ax.set_ylabel("Achieved Mean FPS")
    ax.set_title("Achieved FPS vs Target FPS")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / "fps_vs_target.png", "fps_vs_target")


# ---------------------------------------------------------------------------
# Plot 2: Energy per frame bar chart
# ---------------------------------------------------------------------------

def plot_energy_per_frame(rows: list[dict], out_dir: Path) -> None:
    """Bar chart: energy/frame (mJ) grouped by model, colored by precision."""
    models = sorted({r["model"] for r in rows})
    precisions = sorted({r["precision"] for r in rows})
    prec_colors = {"fp32": "#4C72B0", "fp16": "#DD8452", "bf16": "#55A868"}

    x = np.arange(len(models))
    width = 0.8 / max(len(precisions), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, prec in enumerate(precisions):
        ys = []
        for model in models:
            subset = [
                r for r in rows
                if r["model"] == model
                and r["precision"] == prec
                and r["status"] == "ok"
                and r.get("output_stream", "none") == "none"
            ]
            if subset:
                mj = np.mean([_safe_float(r["energy_per_frame_j"]) * 1000 for r in subset])
            else:
                mj = 0.0
            ys.append(mj)
        offset = (i - len(precisions) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, ys, width * 0.9,
            label=prec, color=prec_colors.get(prec, f"C{i}"),
        )
        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y + 0.2,
                    f"{y:.1f}", ha="center", va="bottom", fontsize=7,
                )

    short_names = [m.replace("_mobilenet_v3_large", "\n_mnv3l").replace("fasterrcnn", "frcnn")
                   for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Energy per frame (mJ)")
    ax.set_title("Energy per Inference Frame by Model × Precision\n(output_stream=none)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir / "energy_per_frame.png", "energy_per_frame")


# ---------------------------------------------------------------------------
# Plot 3: Stage energy stacked bar
# ---------------------------------------------------------------------------

STAGE_COLORS = {
    "capture":     "#4C72B0",
    "preprocess":  "#DD8452",
    "infer":       "#55A868",
    "infer_fused": "#55A868",
    "postprocess": "#C44E52",
    "filter":      "#8172B3",
    "annotate":    "#937860",
    "encode":      "#DA8BC3",
    "idle/other":  "#CCCCCC",
}


def plot_stage_energy_breakdown(rows: list[dict], out_dir: Path) -> None:
    """Stacked bar: stage share of total energy per model (output_stream=none)."""
    models = sorted({r["model"] for r in rows})
    # Collect one representative run per model (fp16 preferred, then fp32)
    model_stage_data: dict[str, dict[str, float]] = {}
    for model in models:
        for prec in ["fp16", "fp32"]:
            subset = [
                r for r in rows
                if r["model"] == model
                and r["precision"] == prec
                and r.get("output_stream", "none") == "none"
                and r["status"] == "ok"
            ]
            if not subset:
                continue
            # Average stage energies across matching runs
            run_stage_energies: dict[str, list[float]] = {}
            total_js: list[float] = []
            for row in subset:
                se = _load_stage_energy(row["run_dir"])
                total_js.append(_safe_float(row.get("energy_total_j", 0)))
                for entry in se:
                    stage = entry["stage"]
                    ej = _safe_float(entry["energy_j"])
                    run_stage_energies.setdefault(stage, []).append(ej)

            avg_total_j = np.mean(total_js) if total_js else 0
            avg_stages = {s: np.mean(vs) for s, vs in run_stage_energies.items()}
            attributed = sum(avg_stages.values())
            idle = max(avg_total_j - attributed, 0.0)
            avg_stages["idle/other"] = idle
            model_stage_data[model] = avg_stages
            break

    if not model_stage_data:
        print("  No stage energy data found; skipping stage_energy_breakdown.png")
        return

    all_stages = sorted(
        {s for d in model_stage_data.values() for s in d},
        key=lambda s: -(sum(d.get(s, 0) for d in model_stage_data.values())),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_stage_data))
    bottoms = np.zeros(len(model_stage_data))

    for stage in all_stages:
        heights = np.array([
            model_stage_data[m].get(stage, 0.0)
            for m in model_stage_data
        ])
        color = STAGE_COLORS.get(stage, f"C{all_stages.index(stage)}")
        ax.bar(x, heights, bottom=bottoms, label=stage, color=color, width=0.6)
        bottoms += heights

    short_names = [
        m.replace("_mobilenet_v3_large", "\n_mnv3l").replace("fasterrcnn", "frcnn")
        for m in model_stage_data
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Energy (J)")
    ax.set_title("Stage Energy Breakdown per Model\n(fp16, output_stream=none)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir / "stage_energy_breakdown.png", "stage_energy_breakdown")


# ---------------------------------------------------------------------------
# Plot 4: Streaming overhead comparison
# ---------------------------------------------------------------------------

def plot_streaming_overhead(rows: list[dict], out_dir: Path) -> None:
    """Bar chart: total energy and per-stage for the 4 streaming modes."""
    streaming_modes = ["none", "mjpeg_cpu", "rtp_h264_sw", "rtp_h264_nvenc"]
    colors = {"none": "#4C72B0", "mjpeg_cpu": "#DD8452",
              "rtp_h264_sw": "#55A868", "rtp_h264_nvenc": "#C44E52"}

    # Find the reference config (yolov8n fp16 or ssdlite fp16)
    ref_models = ["yolov8n", "ssdlite320_mobilenet_v3_large"]
    ref_cfg = None
    for model in ref_models:
        if any(r["model"] == model for r in rows):
            ref_cfg = model
            break
    if ref_cfg is None and rows:
        ref_cfg = rows[0]["model"]

    mode_data: dict[str, dict] = {}
    for mode in streaming_modes:
        subset = [
            r for r in rows
            if r.get("model") == ref_cfg
            and r.get("output_stream") == mode
            and r.get("status") == "ok"
        ]
        if not subset:
            continue
        total_js = [_safe_float(r.get("energy_total_j", 0)) for r in subset]
        epf_mjs = [_safe_float(r.get("energy_per_frame_j", 0)) * 1000 for r in subset]
        mode_data[mode] = {
            "total_j": np.mean(total_js),
            "epf_mj": np.mean(epf_mjs),
        }
        # Load stage energies
        for row in subset[:1]:  # first run's stage breakdown
            se = _load_stage_energy(row["run_dir"])
            stage_j: dict[str, float] = {}
            for entry in se:
                stage = entry["stage"]
                stage_j[stage] = stage_j.get(stage, 0.0) + _safe_float(entry["energy_j"])
            mode_data[mode]["stage_j"] = stage_j

    if len(mode_data) < 2:
        print("  Not enough streaming-mode data; skipping streaming_overhead.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total energy per frame
    ax = axes[0]
    modes_present = [m for m in streaming_modes if m in mode_data]
    epf_vals = [mode_data[m]["epf_mj"] for m in modes_present]
    bars = ax.bar(
        modes_present, epf_vals,
        color=[colors.get(m, "gray") for m in modes_present],
    )
    for bar, v in zip(bars, epf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Energy per frame (mJ)")
    ax.set_title(f"Streaming overhead — {ref_cfg}\n(energy per frame)")
    ax.grid(axis="y", alpha=0.3)

    # Right: stacked stage breakdown
    ax = axes[1]
    all_stages = sorted({
        s for m in modes_present
        for s in mode_data[m].get("stage_j", {})
    })
    bottoms = np.zeros(len(modes_present))
    for stage in all_stages:
        heights = np.array([
            mode_data[m].get("stage_j", {}).get(stage, 0.0)
            for m in modes_present
        ])
        color = STAGE_COLORS.get(stage, f"C{all_stages.index(stage)}")
        ax.bar(modes_present, heights, bottom=bottoms, label=stage, color=color, width=0.6)
        bottoms += heights
    ax.set_ylabel("Energy (J)")
    ax.set_title("Stage breakdown by streaming mode")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / "streaming_overhead.png", "streaming_overhead")


# ---------------------------------------------------------------------------
# Plot 5: Sustained-run timeline
# ---------------------------------------------------------------------------

def plot_sustained_timeline(sweep_dir: Path, out_dir: Path) -> None:
    """Power + FPS over time for sustained runs (20 min)."""
    sustained_dir = sweep_dir / "sustained"
    if not sustained_dir.exists():
        # Also check within the sweep dir itself for runs with long duration
        candidates = list(sweep_dir.glob("*/frames.csv"))
        if not candidates:
            print("  No sustained run data found; skipping sustained_timeline.png")
            return
        # Use the longest run we can find
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        frames_csv = candidates[0]
        run_dir = frames_csv.parent
    else:
        runs = sorted(sustained_dir.glob("*/frames.csv"))
        if not runs:
            print("  No sustained run data in sustained/; skipping sustained_timeline.png")
            return
        frames_csv = runs[0]
        run_dir = frames_csv.parent

    # Load frames.csv
    rows: list[dict] = []
    with frames_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    # Bin into 1-second windows
    t0_ns = _safe_float(rows[0].get("t_capture_start_ns", 0))
    times_s = [(_safe_float(r.get("t_capture_start_ns", 0)) - t0_ns) * 1e-9 for r in rows]
    fps_inst = [_safe_float(r.get("fps_inst", 0)) for r in rows]

    # Load power trace
    power_csv = run_dir / "power_trace.csv"
    power_t: list[float] = []
    power_total: list[float] = []
    if power_csv.exists():
        with power_csv.open() as f:
            for row in csv.DictReader(f):
                if "mono_ns" in row:
                    pt = (_safe_float(row["mono_ns"]) - t0_ns) * 1e-9
                elif "elapsed_ms" in row:
                    pt = _safe_float(row["elapsed_ms"]) / 1000
                else:
                    continue
                total_w = sum(
                    _safe_float(row.get(c, 0))
                    for c in ["cpu_power_mW", "gpu_power_mW", "io_power_mW"]
                ) / 1000.0
                power_t.append(pt)
                power_total.append(total_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(times_s, fps_inst, color="steelblue", linewidth=0.6, alpha=0.8)
    ax1.set_ylabel("Instantaneous FPS")
    ax1.set_title(f"Sustained run: {run_dir.name}")
    ax1.grid(True, alpha=0.3)

    if power_total:
        ax2.plot(power_t, power_total, color="tomato", linewidth=0.6, alpha=0.8)
        ax2.set_ylabel("Total power (W)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir / "sustained_timeline.png", "sustained_timeline")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate camera benchmark plots from a sweep directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sweep-dir",
        required=True,
        help="Path to a sweep output directory containing sweep_summary.csv.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output PNGs (default: <sweep-dir>/plots/).",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", "fps", "energy", "stages", "streaming", "sustained"],
        help="Which plots to generate.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_sweep_csv(sweep_dir)
    want_all = "all" in args.plots

    print(f"Sweep dir : {sweep_dir}")
    print(f"Runs loaded: {len(rows)}")
    print(f"Output dir: {out_dir}")

    if want_all or "fps" in args.plots:
        print("Plot 1: FPS vs target ...")
        plot_fps_vs_target(rows, out_dir)

    if want_all or "energy" in args.plots:
        print("Plot 2: Energy per frame ...")
        plot_energy_per_frame(rows, out_dir)

    if want_all or "stages" in args.plots:
        print("Plot 3: Stage energy breakdown ...")
        plot_stage_energy_breakdown(rows, out_dir)

    if want_all or "streaming" in args.plots:
        print("Plot 4: Streaming overhead ...")
        plot_streaming_overhead(rows, out_dir)

    if want_all or "sustained" in args.plots:
        print("Plot 5: Sustained timeline ...")
        plot_sustained_timeline(sweep_dir, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
