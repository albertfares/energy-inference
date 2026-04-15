"""MJPEG over HTTP streamer — CPU path (libjpeg-turbo via cv2.imencode)."""
from __future__ import annotations

import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .base import OutputStreamer


class MJPEGStreamer(OutputStreamer):
    """Serve annotated frames as MJPEG over HTTP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        quality: int = 80,
    ) -> None:
        self.host = host
        self.port = port
        self.quality = quality
        self._frame_state: dict = {"jpeg": None}
        self._frame_lock = threading.Lock()
        self._server = None

    def open(self) -> None:
        self._server = _build_mjpeg_server(
            self.host, self.port, self._frame_state, self._frame_lock
        )
        print(f"MJPEG stream: http://{self.host}:{self.port}/stream.mjpg")

    def push(self, frame_bgr: np.ndarray) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return
        ok, jpg = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if ok:
            with self._frame_lock:
                self._frame_state["jpeg"] = jpg.tobytes()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _build_mjpeg_server(
    host: str, port: int, frame_state: dict, frame_lock: threading.Lock
) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt: str, *args: object) -> None:  # noqa: ANN002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = (
                    b"<html><head><title>Cam Bench</title></head>"
                    b"<body style='margin:0;background:#111;'>"
                    b"<img src='/stream.mjpg' style='width:100vw;'/>"
                    b"</body></html>"
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream.mjpg":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpeg = frame_state.get("jpeg")
                    if jpeg is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
