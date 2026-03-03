#!/usr/bin/env python3
"""
INA3221 power sampler helper.

Provides a small wrapper around an external sampler binary (e.g. `sample_ina3221`)
and utilities to parse its CSV output and integrate power over time ranges.

Unit tests are included at the bottom of this file and can be run with any of:

    python -m unittest this_file.py
    pytest this_file.py
"""

import argparse  # currently unused, but kept for potential CLI extension
from pathlib import Path
from typing import Optional, Dict, Tuple

import os
import shutil
import subprocess
import time
import logging
import signal
import tempfile
import unittest

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


import ctypes
import os
import sys

# Platform-specific monotonic clock parsing (requires Linux librt for monotonic clock)
try:
    CLOCK_MONOTONIC = 1  # see <linux/time.h>

    class timespec(ctypes.Structure):
        _fields_ = [
            ('tv_sec', ctypes.c_long),
            ('tv_nsec', ctypes.c_long)
        ]

    librt = ctypes.CDLL('librt.so.1', use_errno=True)
    clock_gettime = librt.clock_gettime
    clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(timespec)]

    def monotonic_time():
        t = timespec()
        if clock_gettime(CLOCK_MONOTONIC, ctypes.pointer(t)) != 0:
            errno_ = ctypes.get_errno()
            raise OSError(errno_, os.strerror(errno_))
        return t.tv_sec * 1e9 + t.tv_nsec

except Exception:
    # Fallback for Mac (Darwin) and environments without librt.so.1
    # INA3221 sampler is a Linux-only tool natively anyway, so if missing, we just
    # emulate monotonic time with perf_counter_ns for backward-compatibility logic
    import time
    def monotonic_time():
        return float(time.perf_counter_ns())


class INA3221Sampler:
    """
    Helper to launch a power-sampling process that records INA3221 readings
    to a CSV file, and then post-process that CSV.

    Backward compatible CSV schemas (columns may be a superset):

    (A) Old schema:
        elapsed_ms: float
        cpu_power_mW: float (optional)
        gpu_power_mW: float (optional)
        io_power_mW:  float (optional)

      Timeline: absolute time reconstructed as:
        t_abs = sample_start_perf + elapsed_ms / 1000
      where sample_start_perf is time.perf_counter() captured at start().

    (B) New schema:
        mono_ns: uint64 (CLOCK_MONOTONIC nanoseconds)
        cpu_power_mW: float (optional)
        gpu_power_mW: float (optional)
        io_power_mW:  float (optional)

      Timeline: absolute time (seconds) is:
        t_abs = mono_ns * 1e-9
      which matches a monotonic clock reference (e.g., monotonic_time()).

    NOTE:
      - If you use the new schema, you should pass t0/t1 in the same reference
        (monotonic seconds). In your validation script, that typically means
        capturing t_start/t_end with monotonic_time(), not time.perf_counter().
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        exe_candidate: str = "sample_ina3221",
        hz: int = 1000,
        power_csv: str = "/tmp/power_trace.csv",
        hw: str = "both",
    ):
        """
        Args:
            exe_candidate:
                Sampler executable name or path (e.g. "sample_ina3221" or "./sample_ina3221").
            hz:
                Sampling frequency in Hertz.
            power_csv:
                Output CSV file path where the sampler will write its readings.
            hw:
                Which hardware rails to monitor. Conventionally:
                    "gpu", "cpu", "io", "both" (gpu+cpu), "all" (gpu+cpu+io).
                The exact semantics are implemented in the external sampler.
        """
        self.exe_candidate = exe_candidate
        self.hz = hz
        self.power_csv = power_csv
        self.hw = hw

        self.proc: Optional[subprocess.Popen] = None
        self.sample_start_perf: Optional[float] = None

        # Optional: monotonic timestamp recorded at start() for new schema alignment
        self.sample_start_mono: Optional[float] = None

        # Optional: determined at CSV load time ("elapsed_ms" or "mono_ns")
        self._timebase: Optional[str] = None

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------
    def _find_sampler_exe(self, candidate: str) -> Optional[str]:
        """
        Try to locate the sampler executable.

        Search order:
          1) Same directory as this file.
          2) PATH via shutil.which.
          3) Literal path in `candidate`.
        """
        base_dir = Path(__file__).resolve().parent
        local = (base_dir / candidate).resolve()
        if local.exists() and os.access(str(local), os.X_OK):
            return str(local)

        found = shutil.which(candidate)
        if found:
            return found

        cand = Path(candidate)
        if cand.exists() and os.access(str(cand), os.X_OK):
            return str(cand)

        return None

    def start(self) -> Tuple[Optional[subprocess.Popen], Optional[float]]:
        """
        Launch the external sampler process.

        Returns:
            (proc, t0) where
                proc is the subprocess.Popen object, or None on failure.
                t0   is time.perf_counter() at launch, or None on failure.

        Note:
            For the new CSV schema ("mono_ns"), this timestamp is not used for
            reconstructing the sample timeline, but we keep returning it for
            backward compatibility.
        """
        exe = self._find_sampler_exe(self.exe_candidate)
        if not exe:
            logger.warning("Sampler executable not found: %s", self.exe_candidate)
            self.proc = None
            self.sample_start_perf = None
            self.sample_start_mono = None
            return None, None

        cmd = [
            exe,
            "--hz",
            str(self.hz),
            "--out",
            self.power_csv,
            "--duration-ms",
            "0",  # run until stopped
            "--hw",
            self.hw,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp,  # put child in its own process group
            )
        except Exception as e:
            logger.warning("Failed to launch sampler: %s", e)
            self.proc = None
            self.sample_start_perf = None
            self.sample_start_mono = None
            return None, None

        # Record both clocks:
        # - perf_counter for backward compatibility with elapsed_ms schema
        # - monotonic for convenience/alignment when caller uses mono_ns schema
        self.sample_start_perf = time.perf_counter()
        self.sample_start_mono = monotonic_time() 

        logger.info(
            "Launched INA3221 sampler pid=%s -> %s (hw=%s)",
            self.proc.pid,
            self.power_csv,
            self.hw,
        )
        return self.proc, self.sample_start_perf

    def stop(self, timeout: float = 0.2) -> None:
        """
        Stop the sampler process, if it is running.

        Attempts a graceful SIGINT, then terminate(), then kill() as a last resort.
        Ensures self.proc is cleared on exit.
        """
        proc = self.proc
        if not proc:
            return

        pid = proc.pid
        logger.info("Stopping INA3221 sampler pid=%s", pid)
        try:
            # Send SIGINT to the process group.
            try:
                os.killpg(pid, signal.SIGINT)
            except ProcessLookupError:
                # Process already gone.
                pass

            # Try to wait for a clean exit.
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Try terminate().
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Last resort: kill.
                    proc.kill()
                    proc.wait()
        except Exception as e:
            logger.warning("Error stopping sampler pid=%s: %s", pid, e)
        finally:
            self.proc = None
            self.sample_start_perf = None
            self.sample_start_mono = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_sample_start_perf(self) -> Optional[float]:
        """Return the perf_counter timestamp recorded when start() was called."""
        return self.sample_start_perf

    def get_sample_start_mono(self) -> Optional[float]:
        """Return the monotonic timestamp recorded when start() was called."""
        return self.sample_start_mono

    def get_timebase(self) -> Optional[str]:
        """
        Return the timebase inferred from the CSV ("elapsed_ms" or "mono_ns"),
        or None if load_power_times() has not run yet.
        """
        return self._timebase

    # ------------------------------------------------------------------
    # CSV loading and energy integration
    # ------------------------------------------------------------------
    def load_power_times(
        self,
        sample_start_perf: Optional[float] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Read INA3221 CSV and produce:

            {
              "cpu": (t_abs, p_mW),
              "gpu": (t_abs, p_mW),
              "io":  (t_abs, p_mW),
            }

        Timebase detection (backward compatible):

          - If "mono_ns" exists:
                t_abs = mono_ns * 1e-9
            (seconds in CLOCK_MONOTONIC reference)

          - Else if "elapsed_ms" exists:
                t_abs = sample_start_perf + elapsed_ms / 1000
            (seconds in the same reference as time.perf_counter())

        Args:
            sample_start_perf:
                Required only for old schema ("elapsed_ms").
                If None, we fall back to self.sample_start_perf.

        Raises:
            ValueError if neither mono_ns nor elapsed_ms is present,
            or if no power columns are present.
        """
        if not os.path.exists(self.power_csv):
            raise FileNotFoundError(f"Power CSV not found: {self.power_csv}")

        df = pd.read_csv(self.power_csv)

        # Determine timebase
        if "mono_ns" in df.columns:
            self._timebase = "mono_ns"
            # mono_ns encodes the system-wide monotonic clock in nanoseconds
            t_abs = df["mono_ns"].to_numpy(dtype=np.float64) * 1e-9
        elif "elapsed_ms" in df.columns:
            self._timebase = "elapsed_ms"
            if sample_start_perf is None:
                sample_start_perf = self.sample_start_perf
            if sample_start_perf is None:
                raise RuntimeError(
                    "CSV uses 'elapsed_ms' but sample_start_perf is None; "
                    "call start() before load_power_times() or pass sample_start_perf explicitly."
                )
            elapsed_s = df["elapsed_ms"].to_numpy(dtype=np.float64) / 1000.0
            t_abs = float(sample_start_perf) + elapsed_s  # same reference as time.perf_counter()
        else:
            raise ValueError("Power CSV missing both 'mono_ns' and 'elapsed_ms' columns")

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if "cpu_power_mW" in df.columns:
            result["cpu"] = (
                t_abs,
                df["cpu_power_mW"].to_numpy(dtype=np.float64),
            )

        if "gpu_power_mW" in df.columns:
            result["gpu"] = (
                t_abs,
                df["gpu_power_mW"].to_numpy(dtype=np.float64),
            )

        if "io_power_mW" in df.columns:
            result["io"] = (
                t_abs,
                df["io_power_mW"].to_numpy(dtype=np.float64),
            )

        if not result:
            raise ValueError(
                "No cpu_power_mW / gpu_power_mW / io_power_mW columns in power CSV"
            )

        return result

    def integrate_interval(
        self,
        t_samples: np.ndarray,
        p_samples_mw: np.ndarray,
        t0: float,
        t1: float,
    ) -> float:
        """
        Integrate power between t0 and t1 (seconds) using the trapezoidal rule.

        Args:
            t_samples:
                1D numpy array of timestamps (seconds), same reference as t0/t1.
            p_samples_mw:
                1D numpy array of power samples in milliwatts, aligned with t_samples.
            t0, t1:
                Start and end time in seconds.

        Returns:
            Energy in Joules (float). Returns 0.0 for invalid ranges or if no
            finite power samples are available.
        """
        if t1 <= t0:
            return 0.0

        if t_samples.size == 0 or p_samples_mw.size == 0:
            return 0.0

        valid = np.isfinite(p_samples_mw)
        if not np.any(valid):
            return 0.0

        t_valid = t_samples[valid]
        p_valid = p_samples_mw[valid]

        if t_valid.size == 0:
            return 0.0

        # Interpolate power at the interval boundaries.
        p0 = float(
            np.interp(t0, t_valid, p_valid, left=p_valid[0], right=p_valid[-1])
        )
        p1 = float(
            np.interp(t1, t_valid, p_valid, left=p_valid[0], right=p_valid[-1])
        )

        # Samples strictly inside the interval.
        inside = (t_valid > t0) & (t_valid < t1)
        t_mid = t_valid[inside]
        p_mid = p_valid[inside]

        if t_mid.size == 0:
            # Simple average power over the interval.
            avg_w = (p0 + p1) * 0.5 / 1000.0  # mW -> W
            return avg_w * (t1 - t0)

        times = np.concatenate(([t0], t_mid, [t1]))
        powers_w = np.concatenate(([p0], p_mid, [p1])) / 1000.0  # mW -> W
        return float(np.trapz(powers_w, times))

    def get_energy_range(
        self,
        t0: float,
        t1: float,
        sample_start_perf: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Integrate power between t0 and t1 (seconds) for all available channels.

        Args:
            t0, t1:
                Time range in seconds (must match the CSV timebase):
                  - For old CSV (elapsed_ms): perf_counter seconds.
                  - For new CSV (mono_ns): monotonic seconds (CLOCK_MONOTONIC).
            sample_start_perf:
                Optional override for old CSV only.

        Returns:
            Dictionary mapping channel -> energy in Joules, e.g.:
                { "cpu": energy_J, "gpu": ..., "io": ... }

        Raises:
            Any exceptions raised by load_power_times() for CSV issues.
        """
        result: Dict[str, float] = {}
        power_times = self.load_power_times(sample_start_perf=sample_start_perf)
        for key, (t_samples, p_samples_mw) in power_times.items():
            energy_j = self.integrate_interval(t_samples, p_samples_mw, t0, t1)
            result[key] = energy_j
        return result


# ======================================================================
# Unit tests
# ======================================================================


class TestINA3221Sampler(unittest.TestCase):
    def test_find_sampler_exe_missing(self):
        """_find_sampler_exe should return None for a clearly missing binary."""
        sampler = INA3221Sampler(exe_candidate="definitely_missing_executable_12345")
        path = sampler._find_sampler_exe(sampler.exe_candidate)
        self.assertIsNone(path)

    def test_integrate_interval_zero_for_invalid_range(self):
        """integrate_interval should return 0.0 when t1 <= t0."""
        sampler = INA3221Sampler()
        t_samples = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        p_samples = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)  # 1 W
        e = sampler.integrate_interval(t_samples, p_samples, 2.0, 1.0)
        self.assertEqual(e, 0.0)

    def test_integrate_interval_constant_power(self):
        """
        For constant power of 1000 mW over [0, 2] seconds, energy should be 2 J.
        """
        sampler = INA3221Sampler()
        t_samples = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        p_samples = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)  # 1 W
        e = sampler.integrate_interval(t_samples, p_samples, 0.0, 2.0)
        self.assertAlmostEqual(e, 2.0, places=6)

    def test_stop_without_start_is_noop(self):
        """stop() should not raise if called before start()."""
        sampler = INA3221Sampler()
        try:
            sampler.stop()
        except Exception as e:  # pragma: no cover - defensive
            self.fail(f"stop() raised unexpectedly: {e}")

    def test_load_power_times_and_get_energy_range_old_elapsed_ms(self):
        """
        Old schema: elapsed_ms + sample_start_perf.
        """
        df = pd.DataFrame(
            {
                "elapsed_ms": [0.0, 1000.0, 2000.0],
                "cpu_power_mW": [1000.0, 1000.0, 1000.0],
                "gpu_power_mW": [2000.0, 2000.0, 2000.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name
            df.to_csv(tmp, index=False)

        try:
            sampler = INA3221Sampler(power_csv=csv_path)
            sampler.sample_start_perf = 0.0  # pretend perf_counter start is 0

            power_times = sampler.load_power_times()
            self.assertEqual(sampler.get_timebase(), "elapsed_ms")
            self.assertIn("cpu", power_times)
            self.assertIn("gpu", power_times)

            energies = sampler.get_energy_range(0.0, 2.0)
            self.assertAlmostEqual(energies["cpu"], 2.0, places=5)
            self.assertAlmostEqual(energies["gpu"], 4.0, places=5)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_old_elapsed_ms_raises_without_sample_start(self):
        """
        Old schema requires sample_start_perf either passed or set on the sampler.
        """
        df = pd.DataFrame(
            {
                "elapsed_ms": [0.0, 1000.0],
                "gpu_power_mW": [1000.0, 1000.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name
            df.to_csv(tmp, index=False)

        try:
            sampler = INA3221Sampler(power_csv=csv_path)
            sampler.sample_start_perf = None
            with self.assertRaises(RuntimeError):
                _ = sampler.load_power_times()
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_load_power_times_and_get_energy_range_new_mono_ns(self):
        """
        New schema: mono_ns directly encodes absolute monotonic seconds.
        """
        df = pd.DataFrame(
            {
                "mono_ns": [0, 1_000_000_000, 2_000_000_000],
                "cpu_power_mW": [1000.0, 1000.0, 1000.0],
                "gpu_power_mW": [2000.0, 2000.0, 2000.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name
            df.to_csv(tmp, index=False)

        try:
            sampler = INA3221Sampler(power_csv=csv_path)

            power_times = sampler.load_power_times()  # no need for sample_start_perf
            self.assertEqual(sampler.get_timebase(), "mono_ns")
            self.assertIn("cpu", power_times)
            self.assertIn("gpu", power_times)

            energies = sampler.get_energy_range(0.0, 2.0)
            self.assertAlmostEqual(energies["cpu"], 2.0, places=5)
            self.assertAlmostEqual(energies["gpu"], 4.0, places=5)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_new_mono_ns_ignores_sample_start_perf_even_if_passed(self):
        """
        New schema should ignore sample_start_perf argument; t_abs comes from mono_ns.
        """
        df = pd.DataFrame(
            {
                "mono_ns": [0, 1_000_000_000, 2_000_000_000],
                "gpu_power_mW": [1000.0, 1000.0, 1000.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name
            df.to_csv(tmp, index=False)

        try:
            sampler = INA3221Sampler(power_csv=csv_path)
            power_times = sampler.load_power_times(sample_start_perf=12345.0)
            self.assertEqual(sampler.get_timebase(), "mono_ns")

            t_abs, _ = power_times["gpu"]
            # Must be [0,1,2] seconds, not offset by 12345.
            self.assertTrue(np.allclose(t_abs, np.array([0.0, 1.0, 2.0], dtype=np.float64)))
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_get_sample_start_mono_accessor(self):
        """
        Accessor exists and returns None if start() has not run.
        """
        sampler = INA3221Sampler()
        self.assertIsNone(sampler.get_sample_start_mono())


if __name__ == "__main__":
    # Example usage (manual testing only):
    #
    # This will try to run the external sampler for ~10 seconds.
    # In normal library or test usage, this block is not executed.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Simple INA3221 sampler demo.",
    )
    parser.add_argument(
        "--exe",
        default="./sample_ina3221",
        help="Sampler executable (default: ./sample_ina3221)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=1,
        help="Sampling frequency in Hz (default: 1)",
    )
    parser.add_argument(
        "--power-csv",
        default="/tmp/test.csv",
        help="Output CSV path (default: /tmp/test.csv)",
    )
    parser.add_argument(
        "--hw",
        default="all",
        help='Hardware rails to monitor (default: "all")',
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Sampling duration in seconds for the demo (default: 10)",
    )
    args = parser.parse_args()

    sampler = INA3221Sampler(
        power_csv=args.power_csv,
        hz=args.hz,
        hw=args.hw,
        exe_candidate=args.exe,
    )
    proc, t0 = sampler.start()
    if proc is None:
        logger.error("Failed to start sampler; exiting.")
    else:
        try:
            time.sleep(args.duration)
        finally:
            sampler.stop()
