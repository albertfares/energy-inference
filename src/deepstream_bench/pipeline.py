"""
DeepStream benchmark pipeline.

Builds a GStreamer pipeline:
  v4l2src → (videorate) → nvjpegdec → nvvideoconvert
  → nvstreammux → nvinfer (TensorRT) → fakesink

Frame timestamps are collected via a pad probe on the fakesink sink pad.
Energy is measured by the same INA3221 PowerMonitor used in the PyTorch
pipeline, so the output CSV is directly comparable.

Hardware path vs PyTorch path:
  capture  : NVDEC (nvjpegdec) vs CPU libjpeg-turbo
  preprocess: nvvideoconvert (GPU) vs CPU+CUDA memcpy
  infer    : TensorRT (nvinfer) vs PyTorch CUDA
  postprocess: (fakesink, discarded) vs PyTorch CPU
  encode   : none (always fakesink) vs optional software H.264

No FPS sleep pacing is needed — videorate drops/duplicates frames upstream
to enforce the target rate, so the pipeline naturally runs at that rate.
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# GStreamer imports — only available on Jetson with DeepStream installed.
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    Gst.init(None)
    _GST_AVAILABLE = True
except Exception:
    _GST_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camera_bench.power import PowerMonitor
from camera_bench.metrics import trapz_energy_j


def _require_gst() -> None:
    if not _GST_AVAILABLE:
        raise RuntimeError(
            "GStreamer Python bindings (gi.repository.Gst) not available. "
            "Install python3-gi and the GStreamer DeepStream plugins."
        )


def _hw_jpeg_dec_available() -> bool:
    """Return True if nvjpegdec is available in the GStreamer registry."""
    try:
        reg = Gst.Registry.get()
        return reg.find_plugin("nvjpeg") is not None or reg.find_plugin("nvjpegdec") is not None
    except Exception:
        return False


def _build_pipeline_str(
    device: str,
    width: int,
    height: int,
    target_fps: int,
    nvinfer_config: str,
) -> str:
    """
    Return a Gst.parse_launch-compatible pipeline string.

    The nvstreammux sink pad is connected last so parse_launch can find
    the named element before wiring the upstream chain to mux.sink_0.

    Decode path selection:
      HW (preferred): nvjpegdec — hardware JPEG decoder (NVDEC), NVMM output.
      SW (fallback):  jpegdec + videoconvert + nvvideoconvert — software decode
                      then copy to NVMM.  Used when nvjpegdec is unavailable.
                      The TensorRT inference benefit is identical; only the
                      capture/decode stage reverts to CPU.
    """
    use_hw_dec = _hw_jpeg_dec_available()
    print(f"JPEG decode: {'nvjpegdec (NVDEC hardware)' if use_hw_dec else 'jpegdec (software fallback)'}")

    # Upstream chain (camera → decode → NVMM → mux)
    src_parts = [
        f"v4l2src device={device}",
        f"image/jpeg,width={width},height={height},framerate=30/1",
    ]

    if 0 < target_fps < 30:
        # Drop excess frames to enforce target FPS
        src_parts += [
            "videorate",
            f"image/jpeg,framerate={target_fps}/1",
        ]

    if use_hw_dec:
        src_parts += [
            "nvjpegdec",
            f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height}",
        ]
    else:
        # Software decode → CPU memory → copy to NVMM via nvvideoconvert
        src_parts += [
            "jpegdec",
            f"video/x-raw,format=I420,width={width},height={height}",
            "nvvideoconvert",
            f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height}",
        ]

    src_parts.append("mux.sink_0")
    src_chain = " ! ".join(src_parts)

    # Main chain (mux → nvinfer → fakesink)
    main_chain = (
        f"nvstreammux name=mux batch-size=1 width={width} height={height} "
        f"live-source=1 ! "
        f"nvinfer config-file-path={nvinfer_config} name=infer ! "
        f"fakesink async=false name=sink"
    )

    return f"{main_chain} {src_chain}"


class _FrameCollector:
    """Thread-safe container for frame timestamps collected by the pad probe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timestamps_ns: list[int] = []

    def record(self, ts_ns: int) -> None:
        with self._lock:
            self._timestamps_ns.append(ts_ns)

    def timestamps(self) -> list[int]:
        with self._lock:
            return list(self._timestamps_ns)

    def count(self) -> int:
        with self._lock:
            return len(self._timestamps_ns)


def run_deepstream_benchmark(
    device: str,
    width: int,
    height: int,
    target_fps: int,
    nvinfer_config: str,
    warmup_s: float = 5.0,
    duration_s: float = 60.0,
    sampler_exe: Optional[str] = None,
    ina_hz: int = 1000,
    power_csv_path: Optional[str] = None,
) -> dict:
    """
    Run one DeepStream benchmark run and return a summary dict.

    Parameters
    ----------
    device          : V4L2 device path, e.g. ``/dev/video0``
    width, height   : Camera capture resolution
    target_fps      : 0 = unbounded (run as fast as possible),
                      N > 0 = cap at N FPS via videorate
    nvinfer_config  : Path to the nvinfer .txt config file
    warmup_s        : Seconds to run before starting the INA3221 sampler
    duration_s      : Seconds of timed benchmark after warmup
    sampler_exe     : Path to the ina3221 sampler binary
    ina_hz          : INA3221 sampling rate in Hz
    power_csv_path  : Where to write the raw power trace CSV

    Returns
    -------
    dict with keys:
        n_frames, duration_s, fps_mean, fps_p50, fps_p95,
        energy_total_j, mean_power_w, energy_per_frame_j,
        cpu_rail_j, gpu_rail_j, io_rail_j,
        status, error
    """
    _require_gst()

    collector = _FrameCollector()
    loop = GLib.MainLoop()
    pipeline: Optional[Gst.Pipeline] = None

    # ── Pad probe: count every frame that exits nvinfer ──────────────────────
    def _on_sink_pad_probe(pad, info, _user_data):
        buf = info.get_buffer()
        if buf is not None:
            collector.record(time.monotonic_ns())
        return Gst.PadProbeReturn.OK

    # ── Bus message handler ───────────────────────────────────────────────────
    def _on_bus_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"GStreamer ERROR: {err.message}\n  debug: {dbg}", file=sys.stderr)
            loop.quit()
        return True

    pipeline_str = _build_pipeline_str(device, width, height, target_fps, nvinfer_config)
    print(f"Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as exc:
        return {"status": "failed", "error": f"Pipeline build failed: {exc}"}

    # Attach pad probe on fakesink's sink pad
    sink = pipeline.get_by_name("sink")
    sink_pad = sink.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, _on_sink_pad_probe, None)

    # Attach bus
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _on_bus_message)

    # ── Start pipeline ────────────────────────────────────────────────────────
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        return {"status": "failed", "error": "Pipeline failed to start (PLAYING state)"}

    print(f"Warmup: {warmup_s:.0f}s ...", flush=True)
    time.sleep(warmup_s)

    # ── Start INA3221 power sampler ───────────────────────────────────────────
    power_monitor: Optional[PowerMonitor] = None
    if sampler_exe:
        pcsv = power_csv_path or "/tmp/ds_power_trace.csv"
        power_monitor = PowerMonitor(
            exe_path=sampler_exe,
            hz=ina_hz,
            hw="all",
            csv_path=pcsv,
        )
        ok = power_monitor.start()
        if not ok:
            print("WARNING: INA3221 sampler failed to start; no energy data.", file=sys.stderr)
            power_monitor = None

    print(f"Benchmark: {duration_s:.0f}s ...", flush=True)
    t_start_ns = time.monotonic_ns()
    n_frames_at_start = collector.count()

    # ── Run for duration_s, then send EOS ────────────────────────────────────
    def _stop_callback():
        pipeline.send_event(Gst.Event.new_eos())
        return False  # don't repeat

    GLib.timeout_add(int(duration_s * 1000), _stop_callback)

    loop.run()  # blocks until EOS or error

    t_end_ns = time.monotonic_ns()
    actual_duration_s = (t_end_ns - t_start_ns) * 1e-9

    # ── Stop sampler ──────────────────────────────────────────────────────────
    if power_monitor is not None:
        power_monitor.stop()

    pipeline.set_state(Gst.State.NULL)

    # ── Compute FPS stats ─────────────────────────────────────────────────────
    all_timestamps = collector.timestamps()
    # Keep only timestamps in the timed window [t_start_ns, t_end_ns]
    timed_ts = [ts for ts in all_timestamps if t_start_ns <= ts <= t_end_ns]
    n_frames = len(timed_ts)

    fps_mean = n_frames / max(actual_duration_s, 1e-6)
    if len(timed_ts) >= 2:
        gaps_s = np.diff(timed_ts) * 1e-9
        inst_fps = 1.0 / np.maximum(gaps_s, 1e-6)
        fps_p50 = float(np.percentile(inst_fps, 50))
        fps_p95 = float(np.percentile(inst_fps, 95))
        fps_min = float(inst_fps.min())
        fps_max = float(inst_fps.max())
    else:
        fps_p50 = fps_p95 = fps_min = fps_max = fps_mean

    summary = {
        "n_frames": n_frames,
        "actual_duration_s": round(actual_duration_s, 3),
        "fps_mean": round(fps_mean, 4),
        "fps_p50": round(fps_p50, 4),
        "fps_p95": round(fps_p95, 4),
        "fps_min": round(fps_min, 4),
        "fps_max": round(fps_max, 4),
        "status": "ok",
        "error": "",
    }

    # ── Compute energy ────────────────────────────────────────────────────────
    if power_monitor is not None:
        try:
            power_trace = power_monitor.load_power_trace()
            t0_s = t_start_ns * 1e-9
            t1_s = t_end_ns * 1e-9
            per_rail_j = {
                rail: trapz_energy_j(t_s, p_mw, t0_s, t1_s)
                for rail, (t_s, p_mw) in power_trace.items()
            }
            total_j = sum(per_rail_j.values())
            mean_power_w = total_j / max(actual_duration_s, 1e-6)
            energy_per_frame_j = total_j / max(n_frames, 1)

            summary.update({
                "energy_total_j": round(total_j, 4),
                "mean_power_w": round(mean_power_w, 4),
                "energy_per_frame_j": round(energy_per_frame_j, 6),
                "cpu_rail_j": round(per_rail_j.get("cpu", 0.0), 4),
                "gpu_rail_j": round(per_rail_j.get("gpu", 0.0), 4),
                "io_rail_j": round(per_rail_j.get("io", 0.0), 4),
            })
        except Exception as exc:
            print(f"WARNING: energy computation failed: {exc}", file=sys.stderr)
            summary.update({
                "energy_total_j": None,
                "mean_power_w": None,
                "energy_per_frame_j": None,
                "cpu_rail_j": None,
                "gpu_rail_j": None,
                "io_rail_j": None,
                "energy_error": str(exc),
            })
    else:
        for k in ("energy_total_j", "mean_power_w", "energy_per_frame_j",
                  "cpu_rail_j", "gpu_rail_j", "io_rail_j"):
            summary[k] = None

    return summary
