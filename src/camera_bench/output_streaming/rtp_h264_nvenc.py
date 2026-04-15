"""RTP/H.264 NVENC hardware encoding streamer via GStreamer + PyGObject."""
from __future__ import annotations

import os
import sys

import numpy as np

from .base import OutputStreamer

_DEFAULT_GST_PLUGIN_PATH = "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"


def _ensure_gst_plugin_path() -> None:
    """Set GST_PLUGIN_PATH if unset so conda-forge GStreamer finds NVIDIA plugins."""
    if "GST_PLUGIN_PATH" not in os.environ:
        os.environ["GST_PLUGIN_PATH"] = _DEFAULT_GST_PLUGIN_PATH
        print(
            f"[rtp_h264_nvenc] GST_PLUGIN_PATH unset; using {_DEFAULT_GST_PLUGIN_PATH}",
            file=sys.stderr,
        )


def smoke_test_nvenc() -> bool:
    """
    Run a one-shot GStreamer pipeline to verify nvv4l2h264enc is available.
    Returns True on success, False on any error.

    This must be called before opening the camera so failures are caught early.
    """
    _ensure_gst_plugin_path()
    try:
        import gi  # type: ignore
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore

        Gst.init(None)
        pipeline_str = (
            "videotestsrc num-buffers=1 ! "
            "video/x-raw,format=I420,width=320,height=240 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc ! fakesink"
        )
        pipeline = Gst.parse_launch(pipeline_str)
        pipeline.set_state(Gst.State.PLAYING)
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            5 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        pipeline.set_state(Gst.State.NULL)
        if msg is None:
            print("[rtp_h264_nvenc] smoke test timed out.", file=sys.stderr)
            return False
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[rtp_h264_nvenc] smoke test error: {err}  {dbg}", file=sys.stderr)
            return False
        return True
    except Exception as exc:
        print(f"[rtp_h264_nvenc] smoke test exception: {exc}", file=sys.stderr)
        return False


class RTPH264NvencStreamer(OutputStreamer):
    """
    Stream annotated frames via RTP using the Jetson NVENC hardware block.

    GStreamer pipeline::

        appsrc → videoconvert → nvvidconv → nvv4l2h264enc
               → h264parse → rtph264pay → udpsink

    Requires:
        - PyGObject (conda install -c conda-forge pygobject gst-python)
        - System GStreamer NVIDIA plugins (JetPack)
        - GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
          (set automatically if missing)
    """

    def __init__(
        self,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: float,
        sdp_path: str = "cam_bench_nvenc.sdp",
        bitrate: int = 2_000_000,
    ) -> None:
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = max(float(fps), 1.0)
        self.sdp_path = sdp_path
        self.bitrate = bitrate
        self._pipeline = None
        self._appsrc = None
        self._frame_idx = 0
        self._gst_second: int = 1_000_000_000  # filled at open()

    def open(self) -> None:
        _ensure_gst_plugin_path()
        try:
            import gi  # type: ignore
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyGObject / GStreamer not available. "
                "Install via: conda install -c conda-forge pygobject gst-python"
            ) from exc

        Gst.init(None)
        self._gst_second = int(Gst.SECOND)
        fps_n = int(self.fps)

        pipeline_str = (
            f"appsrc name=src is-live=true block=true format=time ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={fps_n}/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate={self.bitrate} insert-sps-pps=true "
            f"iframeinterval=30 ! "
            f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        self._pipeline.set_state(Gst.State.PLAYING)

        self._write_sdp()
        print(f"RTP/H264-NVENC → rtp://{self.host}:{self.port}  SDP: {self.sdp_path}")
        print(
            f"Receiver (GStreamer): gst-launch-1.0 udpsrc port={self.port} "
            f"caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264\" ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink"
        )

    def push(self, frame_bgr: np.ndarray) -> None:
        if self._appsrc is None:
            return
        try:
            from gi.repository import Gst  # type: ignore
        except ImportError:
            return
        data = bytes(frame_bgr)
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        fps_int = max(int(self.fps), 1)
        pts = int(self._frame_idx * self._gst_second // fps_int)
        buf.pts = pts
        buf.dts = pts
        buf.duration = int(self._gst_second // fps_int)
        self._appsrc.emit("push-buffer", buf)
        self._frame_idx += 1

    def close(self) -> None:
        if self._appsrc is not None:
            try:
                from gi.repository import Gst  # type: ignore  # noqa: F401
                self._appsrc.emit("end-of-stream")
            except Exception:
                pass
        if self._pipeline is not None:
            try:
                from gi.repository import Gst  # type: ignore
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
        self._appsrc = None

    def _write_sdp(self) -> None:
        sdp = (
            "v=0\r\n"
            "o=- 0 0 IN IP4 127.0.0.1\r\n"
            "s=cam_bench_nvenc\r\n"
            f"c=IN IP4 {self.host}\r\n"
            "t=0 0\r\n"
            f"m=video {self.port} RTP/AVP 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
        )
        with open(self.sdp_path, "w") as f:
            f.write(sdp)
