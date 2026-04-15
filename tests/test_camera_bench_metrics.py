"""Unit tests for camera_bench.metrics."""
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure src/ is on the path when running from project root
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camera_bench.metrics import (
    attribute_stage_energy,
    compute_fps_stats,
    compute_stage_latency_stats,
    percentile_stats,
    trapz_energy_j,
)


class TestTrapzEnergyJ(unittest.TestCase):
    def test_constant_1W_over_2s(self):
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([1000.0, 1000.0, 1000.0])  # 1 W
        self.assertAlmostEqual(trapz_energy_j(t, p, 0.0, 2.0), 2.0, places=6)

    def test_constant_2W_over_1s(self):
        t = np.array([0.0, 0.5, 1.0])
        p = np.array([2000.0, 2000.0, 2000.0])
        self.assertAlmostEqual(trapz_energy_j(t, p, 0.0, 1.0), 2.0, places=6)

    def test_zero_for_invalid_range(self):
        t = np.array([0.0, 1.0])
        p = np.array([1000.0, 1000.0])
        self.assertEqual(trapz_energy_j(t, p, 1.0, 0.0), 0.0)

    def test_zero_for_equal_bounds(self):
        t = np.array([0.0, 1.0])
        p = np.array([1000.0, 1000.0])
        self.assertEqual(trapz_energy_j(t, p, 0.5, 0.5), 0.0)

    def test_empty_arrays(self):
        self.assertEqual(trapz_energy_j(np.array([]), np.array([]), 0.0, 1.0), 0.0)

    def test_sub_interval(self):
        # 1 W constant, integrate over [0.5, 1.5] only
        t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        p = np.ones(5) * 1000.0
        self.assertAlmostEqual(trapz_energy_j(t, p, 0.5, 1.5), 1.0, places=6)

    def test_ramp_up(self):
        # Power ramps 0 → 2000 mW over [0, 2] → avg 1 W → energy = 2 J
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([0.0, 1000.0, 2000.0])
        self.assertAlmostEqual(trapz_energy_j(t, p, 0.0, 2.0), 2.0, places=5)

    def test_nan_ignored(self):
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([1000.0, np.nan, 1000.0])
        # Should still integrate the valid samples
        result = trapz_energy_j(t, p, 0.0, 2.0)
        self.assertGreater(result, 0.0)

    def test_window_outside_samples(self):
        # Samples only at [1, 2] but window is [0, 3] → should extrapolate/clamp
        t = np.array([1.0, 2.0])
        p = np.array([1000.0, 1000.0])
        result = trapz_energy_j(t, p, 0.0, 3.0)
        self.assertAlmostEqual(result, 3.0, places=5)


class TestPercentileStats(unittest.TestCase):
    def test_empty(self):
        r = percentile_stats([])
        self.assertEqual(r["mean"], 0.0)

    def test_single(self):
        r = percentile_stats([5.0])
        self.assertAlmostEqual(r["mean"], 5.0)
        self.assertAlmostEqual(r["p50"], 5.0)

    def test_known_values(self):
        r = percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(r["mean"], 3.0)
        self.assertAlmostEqual(r["min"], 1.0)
        self.assertAlmostEqual(r["max"], 5.0)

    def test_numpy_array(self):
        r = percentile_stats(np.array([10.0, 20.0, 30.0]))
        self.assertAlmostEqual(r["mean"], 20.0)


class TestAttributeStageEnergy(unittest.TestCase):
    def test_single_stage(self):
        # 1 W constant over [0, 10]
        t = np.linspace(0, 10, 1000)
        p = np.ones(1000) * 1000.0
        power_data = {"cpu": (t, p)}

        # 100 frames, each with "infer" stage of 10 ms
        intervals = [
            {"stage": "infer", "t_start_s": i * 0.1, "t_end_s": i * 0.1 + 0.01}
            for i in range(10)
        ]
        result = attribute_stage_energy(power_data, intervals)
        self.assertIn("infer", result)
        self.assertIn("cpu", result["infer"])
        # 10 stages × 0.01 s × 1 W ≈ 0.1 J
        self.assertAlmostEqual(result["infer"]["cpu"], 0.1, places=3)

    def test_multi_stage_multi_rail(self):
        t = np.array([0.0, 1.0, 2.0])
        cpu_p = np.array([1000.0, 1000.0, 1000.0])
        gpu_p = np.array([2000.0, 2000.0, 2000.0])
        power_data = {"cpu": (t, cpu_p), "gpu": (t, gpu_p)}

        intervals = [
            {"stage": "capture", "t_start_s": 0.0, "t_end_s": 1.0},
            {"stage": "infer",   "t_start_s": 1.0, "t_end_s": 2.0},
        ]
        result = attribute_stage_energy(power_data, intervals)
        self.assertAlmostEqual(result["capture"]["cpu"], 1.0, places=5)
        self.assertAlmostEqual(result["infer"]["gpu"], 2.0, places=5)

    def test_empty_intervals(self):
        t = np.array([0.0, 1.0])
        p = np.array([1000.0, 1000.0])
        result = attribute_stage_energy({"cpu": (t, p)}, [])
        self.assertEqual(result, {})


class TestComputeFpsStats(unittest.TestCase):
    def test_constant_30fps(self):
        # Frames every 1/30 s
        starts = [int(i * 1e9 / 30) for i in range(100)]
        ends = [s + int(1e7) for s in starts]
        r = compute_fps_stats(starts, ends)
        self.assertAlmostEqual(r["mean"], 30.0, delta=0.1)

    def test_single_frame(self):
        r = compute_fps_stats([0], [1_000_000])
        self.assertEqual(r["mean"], 0.0)

    def test_empty(self):
        r = compute_fps_stats([], [])
        self.assertEqual(r["mean"], 0.0)


class TestComputeStageLattencyStats(unittest.TestCase):
    def test_basic(self):
        # 10 frames each with infer taking 10 ms
        ns_10ms = 10_000_000
        stage_times = [{"infer": (0, ns_10ms)} for _ in range(10)]
        r = compute_stage_latency_stats(stage_times)
        self.assertIn("infer", r)
        self.assertAlmostEqual(r["infer"]["mean"], 10.0, places=3)

    def test_empty(self):
        self.assertEqual(compute_stage_latency_stats([]), {})

    def test_zero_duration_skipped(self):
        # Stages with (0, 0) should not contribute
        stage_times = [
            {"capture": (100, 200), "infer": (0, 0)},
            {"capture": (300, 400), "infer": (0, 0)},
        ]
        r = compute_stage_latency_stats(stage_times)
        self.assertIn("capture", r)
        # infer was always (0,0) so no valid durations → empty stats
        if "infer" in r:
            self.assertEqual(r["infer"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
