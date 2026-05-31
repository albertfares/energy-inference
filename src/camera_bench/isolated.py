"""
Shared harness for the *isolated* per-stage benchmarks.

Background
----------
The original sweep (``camera_bench.pipeline``) runs the WHOLE camera→detect
pipeline and attributes energy to stages by time-aligning the INA3221 trace.
That makes every stage's measurement entangled with the others: capture energy
is measured while the model is also resident, preprocess timing is read out of
a YOLO call that fuses pre+infer+post, etc.

The supervisor's requirement is that each sub-predictor be a *lego brick* —
trained on data from a benchmark that exercises ONLY that stage, with nothing
else running. This module is the common machinery those four benchmarks share:

  A  scripts/bench_capture.py      pure camera loop, no model
  B  scripts/bench_preprocess.py   synthetic frame → resize/normalise/H2D
  C  scripts/bench_inference.py    synthetic input tensor → forward pass
  D  scripts/bench_postprocess.py  synthetic boxes → NMS / decode

Each benchmark wraps the work it wants to measure in a zero-argument callable
and hands it to :func:`measure_loop`, which:

  1. runs ``n_warmup`` untimed iterations (lets clocks/JIT/CUDA settle),
  2. starts the INA3221 sampler,
  3. times ``n_measure`` iterations individually (per-iteration latency),
  4. stops the sampler and integrates the power trace over the timed window,
  5. returns latency percentiles + per-iteration energy (mJ).

Energy convention
-----------------
``energy_mj_per_iter`` is the *total* board energy over the timed window divided
by the iteration count. It therefore INCLUDES the platform baseline power that
was flowing while the stage ran — exactly the same convention the in-pipeline
attribution uses, so the bricks stay comparable with the old decomposed model.
The capture benchmark additionally derives a clean idle/sleep power from its
throttled configs (see scripts/bench_capture.py).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .metrics import percentile_stats, trapz_energy_j
from .power import PowerMonitor


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StageMeasurement:
    """Outcome of one measured configuration."""

    n_iters: int
    latency_ms: dict[str, float]          # mean/p50/p95/min/max
    duration_s: float
    total_energy_j: float
    per_rail_j: dict[str, float]
    energy_mj_per_iter: float
    mean_power_w: float
    extra: dict = field(default_factory=dict)

    def as_row(self, prefix: str) -> dict:
        """Flatten into a CSV-friendly row, prefixing stage-specific keys."""
        row: dict = {
            "n_iters": self.n_iters,
            f"{prefix}_lat_mean_ms": round(self.latency_ms["mean"], 5),
            f"{prefix}_lat_p50_ms":  round(self.latency_ms["p50"], 5),
            f"{prefix}_lat_p95_ms":  round(self.latency_ms["p95"], 5),
            f"{prefix}_lat_min_ms":  round(self.latency_ms["min"], 5),
            f"{prefix}_lat_max_ms":  round(self.latency_ms["max"], 5),
            "duration_s":            round(self.duration_s, 4),
            "total_energy_j":        round(self.total_energy_j, 6),
            f"{prefix}_energy_mj_per_iter": round(self.energy_mj_per_iter, 6),
            "mean_power_w":          round(self.mean_power_w, 4),
        }
        for rail, ej in self.per_rail_j.items():
            row[f"rail_{rail}_j"] = round(ej, 6)
        row.update(self.extra)
        return row


# ---------------------------------------------------------------------------
# Power monitor factory
# ---------------------------------------------------------------------------

def make_power_monitor(
    enable: bool,
    sampler_exe: str,
    csv_path: str,
    hz: int = 1000,
    hw: str = "all",
) -> Optional[PowerMonitor]:
    """Construct a PowerMonitor, or return None when energy is disabled."""
    if not enable:
        return None
    return PowerMonitor(exe_path=sampler_exe, hz=hz, hw=hw, csv_path=csv_path)


# ---------------------------------------------------------------------------
# Core measurement loop
# ---------------------------------------------------------------------------

def measure_loop(
    iter_fn: Callable[[], None],
    n_warmup: int,
    n_measure: int,
    power_monitor: Optional[PowerMonitor] = None,
    sync_fn: Optional[Callable[[], None]] = None,
    target_period_s: float = 0.0,
) -> StageMeasurement:
    """
    Time ``iter_fn`` over ``n_measure`` iterations and attribute board energy.

    Parameters
    ----------
    iter_fn
        Zero-arg callable performing exactly one unit of the stage's work.
    n_warmup
        Untimed iterations run before measurement (and before the sampler
        starts) so caches / CUDA context / camera buffers are primed.
    n_measure
        Timed iterations. Each is wrapped in monotonic_ns timestamps.
    power_monitor
        Started just before the timed window and stopped right after; the full
        trace is integrated over [t_start, t_end]. ``None`` → no energy.
    sync_fn
        Optional barrier (e.g. ``torch.cuda.synchronize``) invoked after each
        timed iteration so GPU work is included in that iteration's latency.
    target_period_s
        Optional FPS-cap emulation. When > 0, after each iteration the loop
        sleeps until ``target_period_s`` has elapsed *since that iteration
        started* (adaptive: sleep = period - work_time). The sleep energy is
        captured in the integrated trace, letting the capture benchmark recover
        idle/sleep power. Compute benchmarks leave it at 0 (run flat-out).

    Returns
    -------
    StageMeasurement
    """
    # ---- Warmup (untimed, no power) ----
    for _ in range(max(0, n_warmup)):
        iter_fn()
    if sync_fn is not None:
        sync_fn()

    # ---- Start power sampler ----
    sampler_ok = False
    if power_monitor is not None:
        sampler_ok = power_monitor.start()
        if not sampler_ok:
            print("WARNING: INA3221 sampler failed to start; "
                  "continuing without energy.", file=sys.stderr)
            power_monitor = None

    # ---- Timed window ----
    latencies_ms: list[float] = []
    t_window_start_ns = time.monotonic_ns()
    for _ in range(n_measure):
        t0 = time.monotonic_ns()
        iter_fn()
        if sync_fn is not None:
            sync_fn()
        t1 = time.monotonic_ns()
        latencies_ms.append((t1 - t0) * 1e-6)
        if target_period_s > 0:
            elapsed_s = (time.monotonic_ns() - t0) * 1e-9
            sleep_s = target_period_s - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)
    t_window_end_ns = time.monotonic_ns()

    # ---- Stop sampler ----
    if power_monitor is not None:
        power_monitor.stop()

    duration_s = (t_window_end_ns - t_window_start_ns) * 1e-9
    lat_stats = percentile_stats(latencies_ms)

    per_rail_j: dict[str, float] = {}
    total_energy_j = 0.0
    if power_monitor is not None:
        try:
            trace = power_monitor.load_power_trace()
            t0_s = t_window_start_ns * 1e-9
            t1_s = t_window_end_ns * 1e-9
            for rail, (t_s, p_mw) in trace.items():
                ej = trapz_energy_j(t_s, p_mw, t0_s, t1_s)
                per_rail_j[rail] = ej
                total_energy_j += ej
        except Exception as exc:
            print(f"WARNING: energy integration failed: {exc}", file=sys.stderr)

    energy_mj_per_iter = (total_energy_j * 1000.0 / n_measure) if n_measure else 0.0
    mean_power_w = total_energy_j / max(duration_s, 1e-9)

    return StageMeasurement(
        n_iters=n_measure,
        latency_ms=lat_stats,
        duration_s=duration_s,
        total_energy_j=total_energy_j,
        per_rail_j=per_rail_j,
        energy_mj_per_iter=energy_mj_per_iter,
        mean_power_w=mean_power_w,
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

class IsolatedCSV:
    """Incremental CSV writer that unions keys across heterogeneous rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []

    def append(self, row: dict) -> None:
        self._rows.append(row)

    def flush(self) -> None:
        import csv
        if not self._rows:
            return
        fields: list[str] = []
        for r in self._rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def torch_cuda_sync_fn(device) -> Optional[Callable[[], None]]:
    """Return a CUDA synchronize callable for ``device`` (or None on CPU)."""
    try:
        import torch
    except ImportError:
        return None
    if getattr(device, "type", "cpu") == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return None
