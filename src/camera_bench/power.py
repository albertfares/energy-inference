"""INA3221 power monitor integration for camera benchmarks."""
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

from energy_inference.tools.INA3221Sampler import INA3221Sampler  # noqa: E402

# Import the same monotonic clock the sampler uses so we can check alignment.
try:
    from energy_inference.tools.INA3221Sampler import monotonic_time as _sampler_monotonic
except ImportError:
    _sampler_monotonic = None  # type: ignore[assignment]


class PowerMonitor:
    """Thin wrapper around INA3221Sampler for single-config camera benchmarks."""

    def __init__(
        self,
        exe_path: str,
        hz: int = 1000,
        hw: str = "all",
        csv_path: Optional[str] = None,
    ) -> None:
        import tempfile

        self.exe_path = exe_path
        self.hz = hz
        self.hw = hw
        self._csv_path: str = csv_path or tempfile.mktemp(suffix="_cam_power.csv")
        self._sampler = INA3221Sampler(
            exe_candidate=exe_path,
            hz=hz,
            power_csv=self._csv_path,
            hw=hw,
        )
        self._proc = None

    @property
    def csv_path(self) -> str:
        return self._csv_path

    def start(self) -> bool:
        """Launch the sampler. Returns True on success."""
        proc, _ = self._sampler.start()
        self._proc = proc
        if proc is None:
            return False
        time.sleep(0.25)  # let sampler write first samples
        return True

    def stop(self) -> None:
        if self._proc is not None:
            self._sampler.stop(timeout=0.5)
            self._proc = None

    def check_clock_alignment_ns(self) -> float:
        """
        Return |offset| in nanoseconds between Python's time.monotonic_ns()
        and the librt CLOCK_MONOTONIC used by the INA3221 sampler.
        Should be < 1_000_000 (1 ms).
        """
        if _sampler_monotonic is None:
            return 0.0
        a = time.monotonic_ns()
        b = int(_sampler_monotonic())
        c = time.monotonic_ns()
        mid = (a + c) // 2
        return float(abs(b - mid))

    def load_power_trace(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {rail: (t_s, p_mw)} for the full recorded window."""
        return self._sampler.load_power_times()

    def get_energy_range(self, t0_s: float, t1_s: float) -> dict[str, float]:
        """Return per-rail energy in Joules over [t0_s, t1_s]."""
        return self._sampler.get_energy_range(t0_s, t1_s)
