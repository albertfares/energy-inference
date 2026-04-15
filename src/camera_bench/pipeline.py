"""Main per-frame benchmark loop with stage timing and energy attribution."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .capture import open_camera, validate_camera_resolution
from .detection import build_annotated_frame, run_staged_detection
from .metrics import (
    attribute_stage_energy,
    compute_fps_stats,
    compute_stage_latency_stats,
    percentile_stats,
    trapz_energy_j,
)
from .models import load_detector
from .output_streaming import get_streamer
from .power import PowerMonitor
from .results import ResultsDir, collect_env_info, make_run_name


def run_benchmark(cfg: dict, results_dir: ResultsDir) -> dict:
    """
    Execute one complete benchmark run.

    Args:
        cfg:         Full configuration dict (from cli.py or sweep.py).
        results_dir: ResultsDir object for writing outputs.

    Returns:
        summary dict (also written to results_dir/summary.json).
    """
    try:
        import cv2  # type: ignore
        import torch
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc

    output_mode = cfg.get("output_stream", "none")

    # NVENC smoke test before touching the camera
    if output_mode == "rtp_h264_nvenc":
        from .output_streaming.rtp_h264_nvenc import smoke_test_nvenc
        if not smoke_test_nvenc():
            raise RuntimeError(
                "NVENC smoke test failed. Verify GStreamer NVIDIA plugins are "
                "installed and GST_PLUGIN_PATH is correct."
            )

    # Device
    use_cpu = cfg.get("cpu", False)
    torch_device = torch.device(
        "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Torch device: {torch_device}")

    # Model
    precision = cfg.get("precision", "fp32")
    detector = load_detector(cfg["model"], torch_device, precision)
    print(f"Model: {detector['name']} ({detector['backend']}, {precision})")
    if detector["backend"] == "ultralytics":
        print(
            "  Note: for YOLO, preprocess/infer/postprocess are fused into "
            "'infer_fused' stage."
        )
    categories = detector["categories"]

    # Camera
    device_idx = cfg.get("device", 0)
    width = cfg.get("width", 640)
    height = cfg.get("height", 480)
    fps_req = cfg.get("fps", 30)
    cap, actual_w, actual_h, actual_fps = open_camera(device_idx, width, height, fps_req)
    validate_camera_resolution(device_idx, actual_w, actual_h, width, height)
    print(f"Camera: /dev/video{device_idx} {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")

    # Streamer
    fps_for_stream = actual_fps if actual_fps > 0 else fps_req
    streamer = get_streamer(
        mode=output_mode,
        width=actual_w,
        height=actual_h,
        fps=fps_for_stream,
        host=cfg.get("stream_host", "127.0.0.1"),
        port=cfg.get("stream_port", 11111),
        bitrate=cfg.get("stream_bitrate", 2_000_000),
        sdp_path=cfg.get("rtp_sdp", "cam_bench.sdp"),
        mjpeg_host=cfg.get("mjpeg_host", "127.0.0.1"),
        mjpeg_port=cfg.get("mjpeg_port", 8080),
    )

    # Power monitor
    enable_power = cfg.get("enable_energy", True)
    power_monitor: Optional[PowerMonitor] = None

    if enable_power:
        power_csv = cfg.get("power_csv") or str(results_dir.path("power_trace_raw.csv"))
        power_monitor = PowerMonitor(
            exe_path=cfg.get("sampler_exe", "src/energy_inference/tools/sample_ina3221"),
            hz=cfg.get("ina_hz", 1000),
            hw=cfg.get("ina_hw", "all"),
            csv_path=power_csv,
        )
        offset_ns = power_monitor.check_clock_alignment_ns()
        suffix = " [WARNING: > 1 ms]" if offset_ns > 1_000_000 else " [OK]"
        print(f"Clock alignment check: offset = {offset_ns / 1e6:.3f} ms{suffix}")

    warmup_frames = cfg.get("warmup_frames", 30)
    duration_s = cfg.get("duration_s", 120)
    target_fps = cfg.get("target_fps", 0)
    score_threshold = cfg.get("score_threshold", 0.45)
    iou_threshold = cfg.get("iou_threshold", 0.45)
    max_detections = cfg.get("max_detections", 8)
    yolo_imgsz = cfg.get("yolo_imgsz", 640)
    stage_energy_on = cfg.get("stage_energy", True)

    # Write config
    full_config = {
        **cfg,
        "actual_width": actual_w,
        "actual_height": actual_h,
        "actual_fps": actual_fps,
        "env": collect_env_info(),
    }
    results_dir.write_config(full_config)

    frame_period_s = 1.0 / target_fps if target_fps > 0 else 0.0

    # Run
    with streamer:
        # Warmup — not timed, not logged, but push through the full pipeline
        # (including streamer) so no pipeline init cost leaks into timed window.
        print(f"Warmup: {warmup_frames} frames ...", flush=True)
        _do_warmup(
            cap, detector, torch_device, precision,
            score_threshold, iou_threshold, max_detections, yolo_imgsz,
            streamer, warmup_frames, cv2,
        )

        # Start power sampler only after warmup
        if power_monitor is not None:
            ok = power_monitor.start()
            if not ok:
                print(
                    "WARNING: INA3221 sampler failed to start; continuing without power data.",
                    file=sys.stderr,
                )
                power_monitor = None

        print(f"Benchmark: {duration_s}s ...", flush=True)
        t_bench_start_ns = time.monotonic_ns()
        prev_cap_start_ns: Optional[int] = None
        frame_rows: list[dict] = []
        frame_stage_intervals: list[dict] = []  # all {stage, t_start_s, t_end_s} entries
        frame_idx = 0

        try:
            while True:
                loop_start_ns = time.monotonic_ns()
                elapsed_s = (loop_start_ns - t_bench_start_ns) * 1e-9
                if elapsed_s >= duration_s:
                    break

                # Capture
                t_cap_start = time.monotonic_ns()
                ok, frame_bgr = cap.read()
                t_cap_end = time.monotonic_ns()
                if not ok:
                    print("Frame read failed, ending benchmark early.", file=sys.stderr)
                    break

                # Detection (staged)
                stage_times, boxes, labels, scores = run_staged_detection(
                    frame_bgr, detector, torch_device, precision,
                    score_threshold, iou_threshold, max_detections, yolo_imgsz, cv2,
                )

                # Annotate + Encode
                t_annotate_start = t_annotate_end = 0
                t_encode_start = t_encode_end = 0
                if streamer.needs_annotation:
                    t_annotate_start = time.monotonic_ns()
                    annotated = build_annotated_frame(
                        frame_bgr, boxes, labels, scores, categories, cv2
                    )
                    t_annotate_end = time.monotonic_ns()

                    t_encode_start = time.monotonic_ns()
                    streamer.push(annotated)
                    t_encode_end = time.monotonic_ns()

                stage_times["capture"] = (t_cap_start, t_cap_end)
                stage_times["annotate"] = (t_annotate_start, t_annotate_end)
                stage_times["encode"] = (t_encode_start, t_encode_end)

                # Total frame latency: capture start → last meaningful stage end
                last_end = max(
                    t_encode_end if t_encode_end > 0 else 0,
                    stage_times.get("filter", (0, 0))[1],
                )
                latency_total_ms = (last_end - t_cap_start) * 1e-6

                # Instantaneous FPS
                if prev_cap_start_ns is not None:
                    gap_s = (t_cap_start - prev_cap_start_ns) * 1e-9
                    fps_inst = 1.0 / max(gap_s, 1e-6)
                else:
                    fps_inst = 0.0
                prev_cap_start_ns = t_cap_start

                # Accumulate stage intervals for energy attribution
                if stage_energy_on and power_monitor is not None:
                    for sname, (ts, te) in stage_times.items():
                        if te > ts:
                            frame_stage_intervals.append({
                                "stage": sname,
                                "t_start_s": ts * 1e-9,
                                "t_end_s": te * 1e-9,
                            })

                frame_rows.append(
                    _build_frame_row(
                        frame_idx, stage_times, len(scores),
                        latency_total_ms, fps_inst,
                    )
                )
                frame_idx += 1

                # FPS pacing
                if frame_period_s > 0:
                    elapsed_frame_s = (time.monotonic_ns() - loop_start_ns) * 1e-9
                    sleep_s = frame_period_s - elapsed_frame_s
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")

        t_bench_end_ns = time.monotonic_ns()

    # Stop sampler (outside the `with streamer` block is fine)
    if power_monitor is not None:
        power_monitor.stop()

    cap.release()

    # Compute summary
    print("Computing summary ...", flush=True)
    n_timed = len(frame_rows)
    actual_duration_s = (t_bench_end_ns - t_bench_start_ns) * 1e-9

    fps_stats = compute_fps_stats(
        [r["t_capture_start_ns"] for r in frame_rows],
        [r["t_capture_end_ns"] for r in frame_rows],
    )

    per_frame_stage_times = [_extract_stage_times_ns(r) for r in frame_rows]
    latency_per_stage = compute_stage_latency_stats(per_frame_stage_times)
    latency_total_stats = percentile_stats([r["latency_total_ms"] for r in frame_rows])

    energy_summary: dict = {}
    stage_energy_rows: list[dict] = []
    power_trace: dict = {}

    if power_monitor is not None and n_timed > 0:
        try:
            power_trace = power_monitor.load_power_trace()
            t0_s = t_bench_start_ns * 1e-9
            t1_s = t_bench_end_ns * 1e-9
            per_rail_j = {
                rail: trapz_energy_j(t_s, p_mw, t0_s, t1_s)
                for rail, (t_s, p_mw) in power_trace.items()
            }
            total_j = sum(per_rail_j.values())
            mean_power_w = total_j / max(actual_duration_s, 1e-6)

            stage_energy: dict[str, dict[str, float]] = {}
            if stage_energy_on and frame_stage_intervals:
                stage_energy = attribute_stage_energy(power_trace, frame_stage_intervals)

            per_stage_j: dict[str, float] = {}
            per_stage_pct: dict[str, float] = {}
            for stage, rail_dict in stage_energy.items():
                ej = sum(rail_dict.values())
                per_stage_j[stage] = round(ej, 6)
                per_stage_pct[stage] = round(100.0 * ej / max(total_j, 1e-9), 2)
                for rail, energy_j in rail_dict.items():
                    stage_energy_rows.append({
                        "stage": stage,
                        "rail": rail,
                        "energy_j": round(energy_j, 6),
                        "mean_power_w": round(
                            energy_j / max(actual_duration_s, 1e-6), 4
                        ),
                        "share_pct": round(
                            100.0 * energy_j / max(total_j, 1e-9), 2
                        ),
                    })

            energy_summary = {
                "total_j": round(total_j, 4),
                "mean_power_w": round(mean_power_w, 4),
                "per_rail_j": {r: round(e, 4) for r, e in per_rail_j.items()},
                "per_stage_j": per_stage_j,
                "per_stage_pct": per_stage_pct,
                "energy_per_frame_j": round(total_j / max(n_timed, 1), 6),
                "energy_per_inference_j": round(total_j / max(n_timed, 1), 6),
            }
        except Exception as exc:
            print(f"WARNING: energy computation failed: {exc}", file=sys.stderr)
            energy_summary = {"error": str(exc)}

    summary = {
        "config": cfg,
        "duration_s": round(actual_duration_s, 3),
        "n_warmup": warmup_frames,
        "n_timed": n_timed,
        "fps": fps_stats,
        "latency_ms": {
            "total": latency_total_stats,
            "per_stage": latency_per_stage,
        },
        "energy": energy_summary,
    }

    # Write outputs
    for row in frame_rows:
        results_dir.append_frame(row)
    results_dir.flush_frames_csv()

    if power_monitor is not None:
        try:
            results_dir.write_power_trace_csv(power_monitor.csv_path)
        except Exception as exc:
            print(f"WARNING: could not copy power trace: {exc}", file=sys.stderr)

    if stage_energy_rows:
        results_dir.write_stage_energy_csv(stage_energy_rows)

    results_dir.write_summary(summary)

    _print_summary(summary, stage_energy_rows)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_warmup(
    cap,
    detector: dict,
    torch_device,
    precision: str,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    yolo_imgsz: int,
    streamer,
    n_warmup: int,
    cv2,
) -> None:
    categories = detector["categories"]
    for _ in range(n_warmup):
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        _, boxes, labels, scores = run_staged_detection(
            frame_bgr, detector, torch_device, precision,
            score_threshold, iou_threshold, max_detections, yolo_imgsz, cv2,
        )
        if streamer.needs_annotation:
            annotated = build_annotated_frame(
                frame_bgr, boxes, labels, scores, categories, cv2
            )
            streamer.push(annotated)


_STAGE_COLS = {
    "capture":     ("t_capture_start_ns",     "t_capture_end_ns"),
    "preprocess":  ("t_preprocess_start_ns",  "t_preprocess_end_ns"),
    "infer":       ("t_infer_start_ns",        "t_infer_end_ns"),
    "infer_fused": ("t_infer_fused_start_ns",  "t_infer_fused_end_ns"),
    "postprocess": ("t_postprocess_start_ns",  "t_postprocess_end_ns"),
    "filter":      ("t_filter_start_ns",       "t_filter_end_ns"),
    "annotate":    ("t_annotate_start_ns",     "t_annotate_end_ns"),
    "encode":      ("t_encode_start_ns",       "t_encode_end_ns"),
}


def _build_frame_row(
    frame_idx: int,
    stage_times: dict[str, tuple[int, int]],
    n_detections: int,
    latency_total_ms: float,
    fps_inst: float,
) -> dict:
    row: dict = {"frame_idx": frame_idx}
    for stage, (sc, ec) in _STAGE_COLS.items():
        ts, te = stage_times.get(stage, (0, 0))
        row[sc] = ts
        row[ec] = te
    row["n_detections"] = n_detections
    row["latency_total_ms"] = round(latency_total_ms, 3)
    row["fps_inst"] = round(fps_inst, 3)
    return row


def _extract_stage_times_ns(row: dict) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for stage, (sc, ec) in _STAGE_COLS.items():
        ts = row.get(sc, 0)
        te = row.get(ec, 0)
        if te > ts:
            result[stage] = (ts, te)
    return result


def _print_summary(summary: dict, stage_energy_rows: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("BENCHMARK SUMMARY")
    print("=" * 62)
    print(f"  Duration : {summary['duration_s']:.1f}s   frames={summary['n_timed']}")
    fps = summary["fps"]
    print(
        f"  FPS      : mean={fps['mean']:.1f}  "
        f"p50={fps['p50']:.1f}  p95={fps['p95']:.1f}  "
        f"min={fps['min']:.1f}  max={fps['max']:.1f}"
    )
    lt = summary["latency_ms"]["total"]
    print(
        f"  Latency  : mean={lt['mean']:.1f}ms  "
        f"p50={lt['p50']:.1f}ms  p95={lt['p95']:.1f}ms"
    )
    e = summary.get("energy", {})
    if e and "total_j" in e:
        print(
            f"  Energy   : total={e['total_j']:.2f}J  "
            f"mean={e['mean_power_w']:.3f}W  "
            f"per_frame={e['energy_per_frame_j'] * 1000:.2f}mJ"
        )
        if e.get("per_rail_j"):
            rails = "  ".join(
                f"{r}={v:.2f}J" for r, v in e["per_rail_j"].items()
            )
            print(f"  Per-rail : {rails}")
        if stage_energy_rows:
            # Aggregate per stage across rails
            stage_totals: dict[str, float] = {}
            for row in stage_energy_rows:
                stage_totals[row["stage"]] = (
                    stage_totals.get(row["stage"], 0.0) + row["energy_j"]
                )
            total_j = e.get("total_j", 1.0)
            print("\n  Stage energy breakdown:")
            for stage, ej in sorted(stage_totals.items(), key=lambda x: -x[1]):
                pct = 100.0 * ej / max(total_j, 1e-9)
                print(f"    {stage:15s}: {ej:.4f} J  ({pct:.1f}%)")
    print("=" * 62)
