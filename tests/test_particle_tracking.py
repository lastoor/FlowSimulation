#!/usr/bin/env python
"""
Unit tests for particle_tracking.py
"""

import sys
import os
import unittest
import warnings
from unittest import mock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import particle_tracking


class TestParticleTracking(unittest.TestCase):
    """Test particle tracking functions."""

    def test_find_le_index(self):
        """Test binary search for lower equal index."""
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Exact matches
        self.assertEqual(particle_tracking.find_le_index(arr, 0.0), 0)
        self.assertEqual(particle_tracking.find_le_index(arr, 2.0), 2)
        self.assertEqual(particle_tracking.find_le_index(arr, 4.0), 3)  # last index n-2? Wait function returns n-2 when val >= arr[n-2]
        # Actually arr[n-2] is 3.0, val=4.0 >= arr[n-2] returns n-2 = 3
        # Interpolation
        self.assertEqual(particle_tracking.find_le_index(arr, 1.5), 1)
        self.assertEqual(particle_tracking.find_le_index(arr, 2.5), 2)
        # Edge cases
        self.assertEqual(particle_tracking.find_le_index(arr, -1.0), 0)  # val <= arr[0]
        self.assertEqual(particle_tracking.find_le_index(arr, 10.0), 3)  # val >= arr[n-2]

    def test_norm2(self):
        """Test L2 norm function."""
        self.assertAlmostEqual(particle_tracking.norm2(3.0, 4.0), 5.0)
        self.assertAlmostEqual(particle_tracking.norm2(0.0, 0.0), 0.0)
        self.assertAlmostEqual(particle_tracking.norm2(-3.0, 4.0), 5.0)

    def test_bilinear_sample(self):
        """Test bilinear interpolation of velocity field."""
        # Create a simple 2x2 grid
        xgrid = np.array([0.0, 1.0])
        ygrid = np.array([0.0, 1.0])
        vx = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # flattened (y,x)
        vy = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        nx = 2
        ny = 2

        # Test at grid points
        vx_val, vy_val = particle_tracking.bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, 0.0, 0.0)
        self.assertAlmostEqual(vx_val, 1.0)
        self.assertAlmostEqual(vy_val, 0.1)

        vx_val, vy_val = particle_tracking.bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, 1.0, 1.0)
        self.assertAlmostEqual(vx_val, 4.0)
        self.assertAlmostEqual(vy_val, 0.4)

        # Test interpolation at center (0.5, 0.5)
        vx_val, vy_val = particle_tracking.bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, 0.5, 0.5)
        expected_vx = (1.0 + 2.0 + 3.0 + 4.0) / 4.0
        expected_vy = (0.1 + 0.2 + 0.3 + 0.4) / 4.0
        self.assertAlmostEqual(vx_val, expected_vx)
        self.assertAlmostEqual(vy_val, expected_vy)

        # Test out of bounds clamping
        vx_val, vy_val = particle_tracking.bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, -1.0, 0.5)
        # Extrapolation for x out of bounds (x = -1.0)
        self.assertAlmostEqual(vx_val, 1.0)

    def test_fastest_trajectory(self):
        """Test selection of fastest trajectory."""
        nP = 5
        max_points = 10
        px = np.zeros((nP, max_points), dtype=np.float32)
        py = np.zeros((nP, max_points), dtype=np.float32)
        pt = np.zeros((nP, max_points), dtype=np.float32)
        L = np.zeros(nP, dtype=np.int32)
        st = np.zeros(nP, dtype=np.int32)

        # Create dummy trajectories
        for i in range(nP):
            L[i] = 3  # each trajectory has 3 points
            pt[i, 0] = 0.0
            pt[i, 1] = i * 1.0  # travel time increases with i
            pt[i, 2] = i * 2.0
            st[i] = 0  # success

        # Make trajectory 2 have smallest travel time (i=2 -> travel time = 2*2.0 = 4.0?)
        # Actually travel time is last time point: pt[i, L[i]-1] = pt[i, 2] = i*2.0
        # So i=0 has travel time 0, i=1 has 2, i=2 has 4, etc.
        # But we need fastest (smallest) travel time: i=0.
        # However, we need at least one successful trajectory with valid travel time.
        # Let's set i=0 travel time = 100, i=1 = 50, i=2 = 75, i=3 = 200, i=4 = 300
        pt[0, 2] = 100.0
        pt[1, 2] = 50.0
        pt[2, 2] = 75.0
        pt[3, 2] = 200.0
        pt[4, 2] = 300.0

        # Also set some trajectories as invalid (status != 0)
        st[3] = 1  # stagnation
        st[4] = 4  # other error

        result = particle_tracking.fastest_trajectory(px, py, pt, L, st)

        # Fastest should be trajectory 1 (index 1) with travel time 50.0
        self.assertEqual(result['particle_id'], 1)
        self.assertAlmostEqual(result['travel_time'], 50.0)
        self.assertEqual(len(result['x']), 3)
        self.assertEqual(len(result['y']), 3)
        self.assertEqual(len(result['t']), 3)
        self.assertTrue(np.all(np.isnan(result['all_travel_times']) == [False, False, False, True, True]))

    def test_fastest_trajectory_no_valid(self):
        """Test fastest_trajectory with no valid trajectories."""
        nP = 3
        max_points = 5
        px = np.zeros((nP, max_points), dtype=np.float32)
        py = np.zeros((nP, max_points), dtype=np.float32)
        pt = np.zeros((nP, max_points), dtype=np.float32)
        L = np.zeros(nP, dtype=np.int32)
        st = np.ones(nP, dtype=np.int32)  # all failed

        with self.assertRaises(RuntimeError):
            particle_tracking.fastest_trajectory(px, py, pt, L, st)

    def test_summarize_arrivals_fraction(self):
        """Arrival summaries should preserve arrival order and threshold selection."""
        travel_times = np.array([5.0, np.nan, 2.0, 7.0], dtype=np.float64)
        status = np.array([0, 1, 0, 0], dtype=np.int32)

        summary = particle_tracking.summarize_arrivals(
            travel_times,
            status,
            stop_mode="fraction",
            arrival_fraction=0.5,
        )

        np.testing.assert_array_equal(summary["arrival_order_ids"], np.array([2, 0, 3], dtype=np.int32))
        np.testing.assert_allclose(summary["arrival_order_times"], np.array([2.0, 5.0, 7.0]))
        np.testing.assert_array_equal(summary["selected_particle_ids"], np.array([2, 0], dtype=np.int32))
        self.assertEqual(summary["stopping_particle_id"], 0)
        self.assertAlmostEqual(summary["stopping_travel_time"], 5.0)
        self.assertTrue(summary["threshold_reached"])

    def test_particle_tracking_cpu_uniform_velocity(self):
        """Test particle tracking with uniform velocity field (should be straight line)."""
        # Create a simple uniform velocity field (vx = 1, vy = 0)
        nx, ny = 10, 10
        x = np.linspace(0, 9, nx, dtype=np.float32)
        y = np.linspace(0, 9, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)  # shape (ny, nx)
        vy = np.zeros((ny, nx), dtype=np.float32)

        # Seed particles along left boundary
        nP = 3
        seeds_x = np.full(nP, x[0], dtype=np.float32)
        seeds_y = np.linspace(y[0], y[-1], nP, dtype=np.float32)

        params = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'right_x': x[-1],  # stop when particle reaches right boundary
            'eps_speed': 1e-16,
            'ds_min': 1e-5,
            'ds_max': 0.5,
            'tol': 1e-4,
            'y_policy': 'stop',
            'max_points': 2000,
            'max_iters': 200000,
            'threads_per_block': 256,
            'dtype': np.float64
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_cpu(params)

        # All particles should succeed (status 0)
        self.assertTrue(np.all(st == 0))
        # Each particle should have at least 2 points
        self.assertTrue(np.all(L > 1))
        # Final x position should be close to right boundary
        for i in range(nP):
            self.assertAlmostEqual(px[i, L[i]-1], x[-1], delta=0.2)
            # y should remain constant (vy = 0)
            self.assertAlmostEqual(py[i, 0], py[i, L[i]-1], delta=0.01)

    def test_particle_tracking_cpu_zero_velocity(self):
        """Test particle tracking with zero velocity (should stagnate)."""
        nx, ny = 5, 5
        x = np.linspace(0, 4, nx, dtype=np.float32)
        y = np.linspace(0, 4, ny, dtype=np.float32)
        vx = np.zeros((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        seeds_x = np.array([x[0]], dtype=np.float32)
        seeds_y = np.array([y[0]], dtype=np.float32)

        params = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'right_x': x[-1],
            'eps_speed': 1e-16,
            'ds_min': 1e-5,
            'ds_max': 0.5,
            'tol': 1e-4,
            'y_policy': 'stop',
            'max_points': 2000,
            'max_iters': 200000,
            'threads_per_block': 256,
            'dtype': np.float64
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_cpu(params)

        # Should stagnate (status 1)
        self.assertEqual(st[0], 1)
        # Only initial point recorded
        self.assertEqual(L[0], 1)

    def test_particle_tracking_cpu_summary_fraction(self):
        """CPU summary API should support fraction stopping without requiring fastest-path extraction."""
        nx, ny = 10, 10
        x = np.linspace(0, 9, nx, dtype=np.float32)
        y = np.linspace(0, 9, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        params = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.zeros(4, dtype=np.float32),
            "seeds_y": np.linspace(y[0], y[-1], 4, dtype=np.float32),
            "right_x": x[-1],
            "tracking_mode": "arrival",
            "stop_mode": "fraction",
            "arrival_fraction": 0.5,
            "save_trajectories": False,
            "eps_speed": 1e-16,
            "ds_min": 1e-5,
            "ds_max": 0.5,
            "tol": 1e-4,
            "y_policy": "stop",
            "max_points": 2000,
            "max_iters": 200000,
        }

        summary = particle_tracking.particle_tracking_summary(params)

        self.assertEqual(summary["backend"], "cpu")
        self.assertEqual(summary["method"], "rk45")
        self.assertEqual(summary["tracking_mode"], "arrival")
        self.assertEqual(summary["n_arrived"], 4)
        self.assertEqual(summary["target_arrivals"], 2)
        self.assertEqual(summary["selected_particle_ids"].size, 2)
        self.assertNotIn("px", summary)
        self.assertNotIn("selected_trajectories", summary)
        self.assertNotIn("fastest", summary)

    def test_particle_tracking_summary_return_fastest_false(self):
        """Summary API should omit fastest when return_fastest is disabled."""
        nx, ny = 8, 8
        x = np.linspace(0, 7, nx, dtype=np.float32)
        y = np.linspace(0, 7, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        summary = particle_tracking.particle_tracking_summary(
            {
                "backend": "cpu",
                "method": "rk45",
                "tracking_mode": "trajectory",
                "stop_mode": "first",
                "save_trajectories": True,
                "return_fastest": False,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "seeds_x": np.zeros(2, dtype=np.float32),
                "seeds_y": np.linspace(y[0], y[-1], 2, dtype=np.float32),
                "right_x": x[-1],
            }
        )

        self.assertNotIn("fastest", summary)
        self.assertIn("selected_trajectories", summary)

    def test_particle_tracking_summary_pollock_cpu(self):
        """Method selector should support Pollock on CPU."""
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)

        summary = particle_tracking.particle_tracking_summary(
            {
                "backend": "cpu",
                "method": "pollock",
                "tracking_mode": "arrival",
                "stop_mode": "first",
                "save_trajectories": False,
                "x_faces": x_faces,
                "y_faces": y_faces,
                "vx": vx,
                "vy": vy,
                "seeds_x": np.array([0.0, 0.5], dtype=np.float32),
                "seeds_y": np.array([0.25, 1.25], dtype=np.float32),
                "right_x": x_faces[-1],
                "max_points": 16,
                "max_steps": 16,
            }
        )

        self.assertEqual(summary["backend"], "cpu")
        self.assertEqual(summary["method"], "pollock")
        self.assertEqual(summary["stopping_particle_id"], 1)
        self.assertAlmostEqual(summary["stopping_travel_time"], 2.5, delta=2e-5)
        self.assertIn("fastest", summary)

    def test_particle_tracking_summary_pollock_gpu_arrival_smoke(self):
        """Method selector should support Pollock on GPU for arrival summaries."""
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)

        try:
            summary = particle_tracking.particle_tracking_summary(
                {
                    "backend": "gpu",
                    "method": "pollock",
                    "tracking_mode": "arrival",
                    "stop_mode": "fraction",
                    "arrival_fraction": 0.5,
                    "save_trajectories": False,
                    "x_faces": x_faces,
                    "y_faces": y_faces,
                    "vx": vx,
                    "vy": vy,
                    "seeds_x": np.array([0.0, 0.5], dtype=np.float32),
                    "seeds_y": np.array([0.25, 1.25], dtype=np.float32),
                    "right_x": x_faces[-1],
                    "max_points": 16,
                    "max_steps": 16,
                }
            )
        except Exception as exc:
            self.skipTest(f"Taichi GPU Pollock summary unavailable in this environment: {exc}")

        self.assertEqual(summary["backend"], "gpu")
        self.assertEqual(summary["method"], "pollock")
        self.assertEqual(summary["target_arrivals"], 1)
        self.assertEqual(summary["stopping_particle_id"], 1)
        self.assertAlmostEqual(summary["stopping_travel_time"], 2.5, delta=2e-5)

    def test_particle_tracking_summary_pollock_gpu_selected_trajectory(self):
        """GPU Pollock should replay selected trajectories when requested."""
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)

        try:
            summary = particle_tracking.particle_tracking_summary(
                {
                    "backend": "gpu",
                    "method": "pollock",
                    "tracking_mode": "trajectory",
                    "stop_mode": "first",
                    "save_trajectories": True,
                    "x_faces": x_faces,
                    "y_faces": y_faces,
                    "vx": vx,
                    "vy": vy,
                    "seeds_x": np.array([0.0, 0.5], dtype=np.float32),
                    "seeds_y": np.array([0.25, 1.25], dtype=np.float32),
                    "right_x": x_faces[-1],
                    "max_points": 16,
                    "max_steps": 16,
                }
            )
        except Exception as exc:
            self.skipTest(f"Taichi GPU Pollock trajectory summary unavailable in this environment: {exc}")

        self.assertEqual(summary["method"], "pollock")
        self.assertIn("fastest", summary)
        self.assertIn("selected_trajectories", summary)
        self.assertEqual(len(summary["selected_trajectories"]), 1)
        self.assertAlmostEqual(summary["fastest"]["x"][-1], x_faces[-1], places=6)

    def test_particle_tracking_summary_default_method_matches_backend(self):
        """Omitting method should preserve backend-specific defaults."""
        nx, ny = 8, 8
        x = np.linspace(0, 7, nx, dtype=np.float32)
        y = np.linspace(0, 7, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        summary_cpu = particle_tracking.particle_tracking_summary(
            {
                "backend": "cpu",
                "tracking_mode": "arrival",
                "stop_mode": "first",
                "save_trajectories": False,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "seeds_x": np.zeros(2, dtype=np.float32),
                "seeds_y": np.linspace(y[0], y[-1], 2, dtype=np.float32),
                "right_x": x[-1],
            }
        )
        self.assertEqual(summary_cpu["method"], "rk45")

        try:
            summary_gpu = particle_tracking.particle_tracking_summary(
                {
                    "backend": "gpu",
                    "tracking_mode": "arrival",
                    "stop_mode": "first",
                    "save_trajectories": False,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "seeds_x": np.zeros(2, dtype=np.float32),
                    "seeds_y": np.linspace(y[0], y[-1], 2, dtype=np.float32),
                    "right_x": x[-1],
                    "dt": 0.1,
                    "max_steps": 200,
                }
            )
        except Exception as exc:
            self.skipTest(f"Taichi GPU summary unavailable in this environment: {exc}")
        self.assertEqual(summary_gpu["method"], "rk4")

    def test_particle_tracking_summary_gpu_arrival_smoke(self):
        """GPU/Taichi summary should return arrival metrics for a simple uniform field."""
        nx, ny = 8, 8
        x = np.linspace(0, 7, nx, dtype=np.float32)
        y = np.linspace(0, 7, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        params = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.zeros(3, dtype=np.float32),
            "seeds_y": np.linspace(y[0], y[-1], 3, dtype=np.float32),
            "right_x": x[-1],
            "backend": "gpu",
            "tracking_mode": "arrival",
            "stop_mode": "first",
            "save_trajectories": False,
            "dt": 0.1,
            "max_steps": 200,
        }

        try:
            summary = particle_tracking.particle_tracking_summary(params)
        except Exception as exc:
            self.skipTest(f"Taichi GPU summary unavailable in this environment: {exc}")

        self.assertEqual(summary["backend"], "gpu")
        self.assertEqual(summary["method"], "rk4")
        self.assertEqual(summary["tracking_mode"], "arrival")
        self.assertEqual(summary["n_arrived"], 3)
        self.assertEqual(summary["stopping_particle_id"], 0)
        self.assertGreater(summary["stopping_travel_time"], 0.0)

    def test_particle_tracking_summary_rk45_gpu_smoke(self):
        """Method selector should support RK45 on GPU when available."""
        nx, ny = 8, 8
        x = np.linspace(0, 7, nx, dtype=np.float32)
        y = np.linspace(0, 7, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        try:
            summary = particle_tracking.particle_tracking_summary(
                {
                    "backend": "gpu",
                    "method": "rk45",
                    "tracking_mode": "arrival",
                    "stop_mode": "first",
                    "save_trajectories": False,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "seeds_x": np.zeros(2, dtype=np.float32),
                    "seeds_y": np.linspace(y[0], y[-1], 2, dtype=np.float32),
                    "right_x": x[-1],
                    "max_points": 512,
                    "max_iters": 4096,
                }
            )
        except Exception as exc:
            self.skipTest(f"RK45 GPU summary unavailable in this environment: {exc}")

        self.assertEqual(summary["backend"], "gpu")
        self.assertEqual(summary["method"], "rk45")
        self.assertGreaterEqual(summary["n_arrived"], 1)

    def test_particle_tracking_summary_pollock_gpu_internal_right_x(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)
        right_x = np.float32(0.8)

        try:
            summary = particle_tracking.particle_tracking_summary(
                {
                    "backend": "gpu",
                    "method": "pollock",
                    "tracking_mode": "arrival",
                    "stop_mode": "first",
                    "save_trajectories": True,
                    "return_fastest": True,
                    "x_faces": x_faces,
                    "y_faces": y_faces,
                    "vx": vx,
                    "vy": vy,
                    "seeds_x": np.array([0.25], dtype=np.float32),
                    "seeds_y": np.array([0.5], dtype=np.float32),
                    "right_x": right_x,
                    "max_points": 8,
                    "max_steps": 8,
                }
            )
        except Exception as exc:
            self.skipTest(f"Pollock GPU summary unavailable in this environment: {exc}")

        self.assertEqual(summary["stopping_particle_id"], 0)
        self.assertAlmostEqual(summary["stopping_travel_time"], right_x - 0.25, delta=2e-5)
        self.assertAlmostEqual(summary["fastest"]["x"][-1], right_x, places=6)
        self.assertAlmostEqual(summary["fastest"]["y"][-1], 0.5, places=6)

    def test_pollock_gpu_warns_when_taichi_gpu_falls_back(self):
        try:
            import taichi  # noqa: F401
            from simulation.particle_tracking import pollock_gpu
        except Exception as exc:
            self.skipTest(f"Taichi unavailable in this environment: {exc}")

        with mock.patch.object(pollock_gpu, "_ti_initialized", False):
            with mock.patch("taichi.init", side_effect=[RuntimeError("gpu unavailable"), None]) as init_mock:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pollock_gpu._ensure_ti_init()

        self.assertEqual(init_mock.call_count, 2)
        self.assertTrue(any("falling back to Taichi CPU backend" in str(w.message) for w in caught))

    def test_pollock_gpu_tracker_rerun_resets_to_seeds(self):
        try:
            from simulation.particle_tracking import pollock_gpu
        except Exception as exc:
            self.skipTest(f"Pollock GPU module unavailable in this environment: {exc}")

        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)
        seeds_x = np.array([0.25], dtype=np.float32)
        seeds_y = np.array([0.5], dtype=np.float32)
        right_x = np.float32(0.8)

        try:
            pollock_gpu._ensure_ti_init()
            tracker_cls = pollock_gpu._make_pollock_tracker()
            tracker = tracker_cls(x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, 8)
            times1, status1 = tracker.run()
            times2, status2 = tracker.run()
        except Exception as exc:
            self.skipTest(f"Pollock GPU tracker unavailable in this environment: {exc}")

        self.assertEqual(status1[0], 0)
        self.assertEqual(status2[0], 0)
        self.assertAlmostEqual(times1[0], right_x - seeds_x[0], delta=2e-5)
        self.assertAlmostEqual(times2[0], right_x - seeds_x[0], delta=2e-5)

    def test_particle_tracking_summary_invalid_method_backend(self):
        """Unsupported method/backend combinations should fail clearly."""
        with self.assertRaises(ValueError):
            particle_tracking.particle_tracking_summary({"backend": "cpu", "method": "rk4"})

    def test_particle_tracking_cpu_face_velocity_uniform_flow(self):
        """Face-centered mode should advect particles rightward under uniform vx."""
        nx, ny = 10, 10
        x = np.linspace(0, 9, nx, dtype=np.float32)
        y = np.linspace(0, 9, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)

        nP = 3
        seeds_x = np.full(nP, x[0], dtype=np.float32)
        seeds_y = np.linspace(y[0], y[-1], nP, dtype=np.float32)

        params = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'right_x': x[-1],
            'velocity_location': 'face',
            'x_cell_centers': x,
            'y_cell_centers': y,
            'eps_speed': 1e-16,
            'ds_min': 1e-5,
            'ds_max': 0.5,
            'tol': 1e-4,
            'y_policy': 'stop',
            'max_points': 2000,
            'max_iters': 200000,
            'dtype': np.float64
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_cpu(params)

        self.assertTrue(np.all(st == 0))
        self.assertTrue(np.all(L > 1))
        for i in range(nP):
            self.assertAlmostEqual(px[i, L[i]-1], x[-1], delta=0.2)
            self.assertAlmostEqual(py[i, 0], py[i, L[i]-1], delta=0.01)

    def test_particle_tracking_cpu_crossing_interpolates_to_right_boundary(self):
        """Accepted steps that overshoot right_x should be clipped to the crossing point."""
        nx, ny = 10, 4
        x = np.linspace(0, 0.9, nx, dtype=np.float32)
        y = np.linspace(0, 0.3, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)
        seeds_x = np.array([x[0]], dtype=np.float32)
        seeds_y = np.array([y[1]], dtype=np.float32)
        right_x = 0.05

        params = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'right_x': right_x,
            'eps_speed': 1e-16,
            'ds_min': 1e-5,
            'ds_max': 0.5,
            'tol': 1e-4,
            'y_policy': 'stop',
            'max_points': 2000,
            'max_iters': 200000,
            'dtype': np.float64
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_cpu(params)

        self.assertEqual(st[0], 0)
        self.assertAlmostEqual(px[0, L[0] - 1], right_x, delta=1e-5)
        self.assertAlmostEqual(py[0, L[0] - 1], seeds_y[0], delta=1e-5)
        self.assertAlmostEqual(pt[0, L[0] - 1], right_x - seeds_x[0], delta=5e-4)

    def test_particle_tracking_cpu_invalid_velocity_location(self):
        nx, ny = 5, 5
        x = np.linspace(0, 4, nx, dtype=np.float32)
        y = np.linspace(0, 4, ny, dtype=np.float32)
        vx = np.ones((ny, nx), dtype=np.float32)
        vy = np.zeros((ny, nx), dtype=np.float32)
        seeds_x = np.array([x[0]], dtype=np.float32)
        seeds_y = np.array([y[0]], dtype=np.float32)

        params = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'right_x': x[-1],
            'velocity_location': 'bad_mode',
        }
        with self.assertRaises(ValueError):
            particle_tracking.particle_tracking_cpu(params)

    def test_pollock_step_uniform_flow_exits_right_face(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)

        x_new, y_new, dt, st = particle_tracking.pollock_step(
            0.25, 0.5, x_faces, y_faces, vx, vy
        )

        self.assertEqual(st, 0)
        self.assertAlmostEqual(x_new, 1.0, places=6)
        self.assertAlmostEqual(y_new, 0.5, places=6)
        self.assertAlmostEqual(dt, 0.75, places=6)

    def test_trace_pathline_uniform_flow(self):
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)

        path, times, st = particle_tracking.trace_pathline(
            0.25, 0.5, x_faces, y_faces, vx, vy, max_steps=8
        )

        self.assertEqual(st, 0)
        self.assertGreaterEqual(path.shape[0], 2)
        self.assertTrue(np.all(np.diff(path[:, 0]) >= -1e-6))
        self.assertTrue(np.allclose(path[:, 1], 0.5, atol=1e-6))
        self.assertTrue(np.all(np.diff(times) >= -1e-6))
        self.assertAlmostEqual(path[-1, 0], x_faces[-1], places=6)
        self.assertAlmostEqual(times[-1], 2.75, delta=2e-5)

    def test_trace_pathline_large_coordinate_face_crossing_does_not_stagnate(self):
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([10000.0, 20000.0, 30000.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = -20000.0 * np.ones((3, 3), dtype=np.float32)

        path, times, st = particle_tracking.trace_pathline(
            0.25, 25000.0, x_faces, y_faces, vx, vy, max_steps=8
        )

        self.assertEqual(st, 0)
        self.assertGreaterEqual(path.shape[0], 3)
        self.assertTrue(np.all(np.diff(times) >= -1e-6))
        self.assertAlmostEqual(path[1, 1], 20000.0, places=3)
        self.assertLess(path[2, 1], 20000.0)

    def test_particle_tracking_pollock_uniform_flow(self):
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)
        seeds_x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        seeds_y = np.array([0.25, 1.0, 1.75], dtype=np.float32)

        params = {
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx,
            "vy": vy,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": x_faces[-1],
            "max_points": 16,
            "max_steps": 16,
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)

        self.assertTrue(np.all(st == 0))
        self.assertTrue(np.all(L > 1))
        for i in range(seeds_x.size):
            self.assertAlmostEqual(px[i, L[i] - 1], x_faces[-1], places=6)
            self.assertAlmostEqual(py[i, 0], py[i, L[i] - 1], places=6)
            self.assertAlmostEqual(pt[i, L[i] - 1], x_faces[-1] - seeds_x[i], delta=2e-5)

    def test_particle_tracking_pollock_stops_exactly_at_internal_right_x(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)
        right_x = np.float32(0.8)

        params = {
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.array([0.25], dtype=np.float32),
            "seeds_y": np.array([0.5], dtype=np.float32),
            "right_x": right_x,
            "max_points": 8,
            "max_steps": 8,
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)

        self.assertEqual(st[0], 0)
        self.assertEqual(L[0], 2)
        self.assertAlmostEqual(px[0, 1], right_x, places=6)
        self.assertAlmostEqual(py[0, 1], 0.5, places=6)
        self.assertAlmostEqual(pt[0, 1], right_x - 0.25, delta=2e-5)

    def test_particle_tracking_pollock_zero_velocity_stagnates(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.zeros((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)

        params = {
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.array([0.5], dtype=np.float32),
            "seeds_y": np.array([0.5], dtype=np.float32),
            "max_points": 8,
            "max_steps": 8,
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)

        self.assertEqual(st[0], 1)
        self.assertEqual(L[0], 1)
        self.assertAlmostEqual(px[0, 0], 0.5, places=6)
        self.assertAlmostEqual(py[0, 0], 0.5, places=6)
        self.assertAlmostEqual(pt[0, 0], 0.0, places=6)

    def test_particle_tracking_pollock_near_zero_velocity_stagnates(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.full((2, 3), 1e-20, dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)

        params = {
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.array([0.5], dtype=np.float32),
            "seeds_y": np.array([0.5], dtype=np.float32),
            "max_points": 8,
            "max_steps": 8,
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)

        self.assertEqual(st[0], 1)
        self.assertEqual(L[0], 1)
        self.assertAlmostEqual(px[0, 0], 0.5, places=6)
        self.assertAlmostEqual(py[0, 0], 0.5, places=6)

    def test_particle_tracking_pollock_invalid_shapes(self):
        params = {
            "x_faces": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "y_faces": np.array([0.0, 1.0], dtype=np.float32),
            "vx": np.ones((1, 2), dtype=np.float32),
            "vy": np.zeros((2, 2), dtype=np.float32),
            "seeds_x": np.array([0.0], dtype=np.float32),
            "seeds_y": np.array([0.0], dtype=np.float32),
        }

        with self.assertRaises(ValueError):
            particle_tracking.particle_tracking_pollock(params)

    def test_particle_tracking_pollock_fastest_trajectory_compatibility(self):
        x_faces = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 4), dtype=np.float32)
        vy = np.zeros((3, 3), dtype=np.float32)

        params = {
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx,
            "vy": vy,
            "seeds_x": np.array([0.0, 0.5], dtype=np.float32),
            "seeds_y": np.array([0.25, 1.25], dtype=np.float32),
            "right_x": x_faces[-1],
            "max_points": 16,
            "max_steps": 16,
        }

        px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)
        res = particle_tracking.fastest_trajectory(px, py, pt, L, st)

        self.assertEqual(res["particle_id"], 1)
        self.assertAlmostEqual(res["travel_time"], 2.5, delta=2e-5)
        self.assertAlmostEqual(res["x"][-1], x_faces[-1], places=6)

    def test_particle_tracking_pollock_summary_arrival_mode_skips_dense_arrays(self):
        x_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y_faces = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        vx = np.ones((2, 3), dtype=np.float32)
        vy = np.zeros((3, 2), dtype=np.float32)

        result = particle_tracking.particle_tracking_summary(
            {
                "backend": "cpu",
                "method": "pollock",
                "tracking_mode": "arrival",
                "stop_mode": "first",
                "save_trajectories": False,
                "return_fastest": True,
                "x_faces": x_faces,
                "y_faces": y_faces,
                "vx": vx,
                "vy": vy,
                "seeds_x": np.array([0.0, 0.5], dtype=np.float32),
                "seeds_y": np.array([0.25, 1.25], dtype=np.float32),
                "right_x": x_faces[-1],
                "max_points": 8,
                "max_steps": 8,
            }
        )

        self.assertNotIn("px", result)
        self.assertNotIn("py", result)
        self.assertNotIn("pt", result)
        self.assertEqual(result["stopping_particle_id"], 1)
        self.assertAlmostEqual(result["stopping_travel_time"], 1.5, delta=2e-5)
        self.assertEqual(result["fastest"]["particle_id"], 1)
        self.assertAlmostEqual(result["fastest"]["x"][-1], x_faces[-1], places=6)


if __name__ == '__main__':
    unittest.main()
