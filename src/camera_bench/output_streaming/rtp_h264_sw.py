"""RTP/H.264 software encoding streamer — ffmpeg + libx264 (CPU)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from .base import OutputStreamer


class RTPH264SWStreamer(OutputStreamer):
    """Stream annotated frames via RTP using ffmpeg's libx264."""

    def __init__(
        self,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: float,
        sdp_path: str = "cam_bench_sw.sdp",
        bitrate: int = 2_000_000,
    ) -> None:
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = max(float(fps), 1.0)
        self.sdp_path = sdp_path
        self.bitrate = bitrate
        self._proc: subprocess.Popen | None = None

    def open(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.2f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-b:v", str(self.bitrate),
            "-g", "30", "-keyint_min", "30",
            "-x264-params", "repeat-headers=1:scenecut=0",
            "-pix_fmt", "yuv420p",
            "-f", "rtp",
            "-sdp_file", self.sdp_path,
            "-payload_type", "96",
            f"rtp://{self.host}:{self.port}?pkt_size=1200",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg to use --output-stream rtp_h264_sw."
            ) from exc
        print(f"RTP/H264-SW → rtp://{self.host}:{self.port}  SDP: {self.sdp_path}")
        print(
            f"Receiver: ffplay -protocol_whitelist file,udp,rtp -i {self.sdp_path}"
        )

    def push(self, frame_bgr: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        if self._proc.poll() is not None:
            print("ffmpeg RTP streamer exited unexpectedly.", file=sys.stderr)
            return
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            print("RTP sw streamer pipe closed.", file=sys.stderr)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
