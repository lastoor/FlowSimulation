#!/usr/bin/env python
"""
Unit tests for particle_tracking.py
"""

import sys
import os
import unittest
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


if __name__ == '__main__':
    unittest.main()
