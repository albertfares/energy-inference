"""No-op streamer — annotation drawing and encoding are skipped entirely."""
from __future__ import annotations

import numpy as np

from .base import OutputStreamer


class NoneStreamer(OutputStreamer):
    """Skips annotation drawing and encoding entirely (lowest-overhead baseline)."""

    @property
    def needs_annotation(self) -> bool:
        return False

    def push(self, frame_bgr: np.ndarray) -> None:  # noqa: ARG002
        pass
