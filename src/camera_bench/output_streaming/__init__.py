"""Output streaming modes for the camera benchmark."""
from __future__ import annotations

from .base import OutputStreamer
from .mjpeg_cpu import MJPEGStreamer
from .none_stream import NoneStreamer
from .rtp_h264_sw import RTPH264SWStreamer

__all__ = ["OutputStreamer", "NoneStreamer", "MJPEGStreamer", "RTPH264SWStreamer", "get_streamer"]


def get_streamer(
    mode: str,
    width: int,
    height: int,
    fps: float,
    host: str = "127.0.0.1",
    port: int = 11111,
    bitrate: int = 2_000_000,
    sdp_path: str = "cam_bench.sdp",
    mjpeg_host: str = "127.0.0.1",
    mjpeg_port: int = 8080,
) -> OutputStreamer:
    """Factory: return an OutputStreamer for the given mode."""
    if mode == "none":
        return NoneStreamer()
    if mode == "mjpeg_cpu":
        return MJPEGStreamer(host=mjpeg_host, port=mjpeg_port)
    if mode == "rtp_h264_sw":
        return RTPH264SWStreamer(
            host=host, port=port, width=width, height=height,
            fps=fps, sdp_path=sdp_path, bitrate=bitrate,
        )
    if mode == "rtp_h264_nvenc":
        from .rtp_h264_nvenc import RTPH264NvencStreamer
        return RTPH264NvencStreamer(
            host=host, port=port, width=width, height=height,
            fps=fps, sdp_path=sdp_path, bitrate=bitrate,
        )
    raise ValueError(
        f"Unknown output stream mode: {mode!r}. "
        "Choose from: none, mjpeg_cpu, rtp_h264_sw, rtp_h264_nvenc"
    )
