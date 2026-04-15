"""Pure metric computation — no I/O, no torch dependency."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def trapz_energy_j(
    t_s: np.ndarray,
    p_mw: np.ndarray,
    t0_s: float,
    t1_s: float,
) -> float:
    """
    Integrate power over [t0_s, t1_s] (seconds) using the trapezoidal rule.

    Args:
        t_s:   1-D array of timestamps in seconds (CLOCK_MONOTONIC reference).
        p_mw:  1-D array of power in milliwatts, aligned with t_s.
        t0_s:  Start of integration window (seconds).
        t1_s:  End of integration window (seconds).

    Returns:
        Energy in Joules.
    """
    if t1_s <= t0_s:
        return 0.0
    if t_s.size == 0 or p_mw.size == 0:
        return 0.0

    valid = np.isfinite(p_mw)
    if not np.any(valid):
        return 0.0
    t_v = t_s[valid]
    p_v = p_mw[valid]

    p0 = float(np.interp(t0_s, t_v, p_v, left=p_v[0], right=p_v[-1]))
    p1 = float(np.interp(t1_s, t_v, p_v, left=p_v[0], right=p_v[-1]))

    inside = (t_v > t0_s) & (t_v < t1_s)
    times = np.concatenate(([t0_s], t_v[inside], [t1_s]))
    powers_w = np.concatenate(([p0], p_v[inside], [p1])) / 1000.0  # mW → W
    return float(np.trapz(powers_w, times))


def percentile_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Return {mean, p50, p95, min, max} for a sequence of floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def attribute_stage_energy(
    power_data: dict[str, tuple[np.ndarray, np.ndarray]],
    stage_intervals: list[dict],
) -> dict[str, dict[str, float]]:
    """
    Attribute energy to pipeline stages by time-aligning the power trace.

    Args:
        power_data:      {rail: (t_s, p_mw)} from INA3221Sampler.load_power_times().
        stage_intervals: List of {stage, t_start_s, t_end_s} dicts (all timed frames).

    Returns:
        {stage: {rail: energy_j, ...}, ...}
    """
    stage_names: set[str] = {s["stage"] for s in stage_intervals}
    result: dict[str, dict[str, float]] = {s: {} for s in stage_names}

    for rail, (t_s, p_mw) in power_data.items():
        accum: dict[str, float] = {s: 0.0 for s in stage_names}
        for iv in stage_intervals:
            stage = iv["stage"]
            t0 = iv["t_start_s"]
            t1 = iv["t_end_s"]
            if t1 > t0:
                accum[stage] += trapz_energy_j(t_s, p_mw, t0, t1)
        for stage in stage_names:
            result[stage][rail] = accum[stage]

    return result


def compute_fps_stats(
    frame_t_start_ns: list[int],
    frame_t_end_ns: list[int],  # noqa: ARG001 — kept for API symmetry
) -> dict[str, float]:
    """
    Compute FPS statistics from per-frame capture-start timestamps.

    Instantaneous FPS_i = 1 / (t_start[i] - t_start[i-1]).
    """
    if len(frame_t_start_ns) < 2:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    starts = np.array(frame_t_start_ns, dtype=np.float64)
    gaps_s = np.diff(starts) * 1e-9
    fps_inst = 1.0 / np.where(gaps_s > 0, gaps_s, 1.0)
    return percentile_stats(fps_inst)


def compute_stage_latency_stats(
    stage_times_ns: list[dict[str, tuple[int, int]]],
) -> dict[str, dict[str, float]]:
    """
    Compute per-stage latency stats (mean/p50/p95/min/max in ms).

    Args:
        stage_times_ns: list of per-frame {stage: (t_start_ns, t_end_ns)}.
    """
    if not stage_times_ns:
        return {}
    stages = list(stage_times_ns[0].keys())
    result: dict[str, dict[str, float]] = {}
    for stage in stages:
        durations_ms = [
            (f[stage][1] - f[stage][0]) * 1e-6
            for f in stage_times_ns
            if stage in f and f[stage][1] > f[stage][0]
        ]
        result[stage] = percentile_stats(durations_ms)
    return result
