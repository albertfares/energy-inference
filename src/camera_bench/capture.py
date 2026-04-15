"""Webcam capture helpers for the camera benchmark."""
from __future__ import annotations

import subprocess
import sys


def open_camera(
    device_idx: int,
    width: int,
    height: int,
    fps: int,
):
    """Open a V4L2 camera and configure resolution + FPS.

    Returns:
        (cap, actual_width, actual_height, actual_fps)

    Raises:
        RuntimeError if the camera cannot be opened.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required.") from exc

    cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open /dev/video{device_idx}. "
            "Check camera device and permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_w, actual_h, actual_fps


def list_supported_modes(device_idx: int) -> list[str]:
    """Use v4l2-ctl to list supported modes (best-effort; returns [] on failure)."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-formats-ext", "-d", f"/dev/video{device_idx}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.splitlines()
        return []
    except Exception:
        return []


def validate_camera_resolution(
    device_idx: int,
    actual_width: int,
    actual_height: int,
    requested_width: int,
    requested_height: int,
) -> None:
    """Warn (stderr) if the camera silently fell back to a different resolution."""
    if actual_width != requested_width or actual_height != requested_height:
        print(
            f"WARNING: Requested {requested_width}x{requested_height} but camera "
            f"/dev/video{device_idx} opened at {actual_width}x{actual_height}. "
            "Results will reflect the actual resolution.",
            file=sys.stderr,
        )
