"""Abstract base class for output streamers."""
from __future__ import annotations

import abc

import numpy as np


class OutputStreamer(abc.ABC):
    """Interface for camera benchmark output streamers."""

    @property
    def needs_annotation(self) -> bool:
        """Return False if annotation drawing (and this push) should be skipped entirely."""
        return True

    def open(self) -> None:
        """Set up the output pipeline (called before warmup)."""

    @abc.abstractmethod
    def push(self, frame_bgr: np.ndarray) -> None:
        """Push one annotated BGR frame to the output."""

    def close(self) -> None:
        """Tear down the output pipeline."""

    def __enter__(self) -> "OutputStreamer":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
