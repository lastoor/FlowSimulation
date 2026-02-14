#!/usr/bin/env python
"""
Unit tests for hydraulic.py
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if petsc4py is available
try:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False

from simulation import hydraulic


class TestHydraulic(unittest.TestCase):
    """Test hydraulic functions."""

    def test_compute_velocity_uniform(self):
        """Test velocity computation for uniform head field."""
        # Create a uniform conductivity field
        K = np.ones((5, 5), dtype=float)
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)

        # Linear head gradient in x-direction
        h = np.tile(x, (5, 1))  # h varies linearly with x, constant in y

        vx, vy = hydraulic.compute_velocity(K, x, y, h)

        # Analytical solution: vx = -K * dh/dx, dh/dx = 1 (since x spacing 1)
        # Actually dh/dx = (x[1]-x[0]) = 1? x spacing is 1, h = x, so dh/dx = 1
        expected_vx = -K * 1.0
        np.testing.assert_array_almost_equal(vx, expected_vx)
        # vy should be zero (no gradient in y)
        np.testing.assert_array_almost_equal(vy, 0.0)

    def test_compute_velocity_constant_head(self):
        """Test velocity computation for constant head (zero gradient)."""
        K = np.ones((3, 3), dtype=float)
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 2, 3)
        h = np.ones((3, 3)) * 5.0  # constant head

        vx, vy = hydraulic.compute_velocity(K, x, y, h)

        # Velocity should be zero everywhere
        np.testing.assert_array_almost_equal(vx, 0.0)
        np.testing.assert_array_almost_equal(vy, 0.0)

    def test_compute_velocity_varying_conductivity(self):
        """Test velocity computation with spatially varying conductivity."""
        K = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        # Linear head gradient in x
        h = np.array([[0.0, 1.0], [0.0, 1.0]])

        vx, vy = hydraulic.compute_velocity(K, x, y, h)

        # dh/dx = 1, so vx = -K * 1
        expected_vx = -K
        np.testing.assert_array_almost_equal(vx, expected_vx)
        np.testing.assert_array_almost_equal(vy, 0.0)

    def test_compute_velocity_boundary_derivatives(self):
        """Test velocity computation at boundaries (one-sided derivatives)."""
        # Use a larger grid to test boundary derivatives
        nx, ny = 5, 5
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 4, ny)
        # Quadratic head field h = x^2
        X, Y = np.meshgrid(x, y)
        h = X**2

        vx, vy = hydraulic.compute_velocity(K, x, y, h)

        # Analytical derivative dh/dx = 2x
        expected_vx = -K * 2 * X
        # At boundaries, finite difference approximation uses one-sided differences
        # Check interior points (columns 1 to nx-2) match well
        interior_slice = (slice(None), slice(1, -1))
        np.testing.assert_array_almost_equal(vx[interior_slice], expected_vx[interior_slice], decimal=2)
        # Check that boundaries are computed (no crash)
        self.assertTrue(np.all(np.isfinite(vx)))

    @unittest.skipIf(not PETSC_AVAILABLE, "petsc4py not available")
    def test_solve_head_uniform_conductivity(self):
        """Test head solver for uniform conductivity (should be linear)."""
        K = np.ones((5, 5), dtype=float)
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)

        h, msg, is_successful = hydraulic.solve_head(K, x, y, dh=0.01)

        self.assertTrue(is_successful, msg=f"Solver failed: {msg}")
        self.assertEqual(h.shape, (5, 5))
        # Head should decrease linearly from left to right
        # Left boundary h = dh * x_len, right boundary h = 0
        x_len = x[-1] - x[0]
        for i in range(5):
            # Each column should have constant head (no flow in y)
            np.testing.assert_array_almost_equal(h[:, i], h[0, i])
        # Check linear gradient (allow small numerical error)
        for i in range(5):
            expected = 0.01 * x_len * (1 - i/4)  # linear from left to right
            self.assertAlmostEqual(h[0, i], expected, delta=0.005)

    @unittest.skipIf(not PETSC_AVAILABLE, "petsc4py not available")
    def test_solve_head_heterogeneous_conductivity(self):
        """Test head solver with heterogeneous conductivity."""
        # Create a conductivity field with a high conductivity channel
        K = np.ones((6, 6), dtype=float)
        K[:, 2:4] = 10.0  # higher conductivity in middle columns

        x = np.linspace(0, 5, 6)
        y = np.linspace(0, 5, 6)

        h, msg, is_successful = hydraulic.solve_head(K, x, y, dh=0.01)

        self.assertTrue(is_successful, msg=f"Solver failed: {msg}")
        self.assertEqual(h.shape, (6, 6))
        # Head should be continuous; no easy analytical solution
        # Just check that head decreases monotonically from left to right
        for row in range(6):
            for col in range(1, 6):
                self.assertLess(h[row, col], h[row, col-1] + 1e-10)

    @unittest.skipIf(not PETSC_AVAILABLE, "petsc4py not available")
    def test_solve_head_zero_conductivity(self):
        """Test head solver with zero conductivity (should still converge?)."""
        K = np.ones((3, 3), dtype=float)
        K[1, 1] = 0.0  # blocked cell

        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 2, 3)

        h, msg, is_successful = hydraulic.solve_head(K, x, y, dh=0.01)

        # Solver may still converge depending on boundary conditions
        # Just ensure it doesn't crash
        self.assertEqual(h.shape, (3, 3))

    @unittest.skipIf(not PETSC_AVAILABLE, "petsc4py not available")
    def test_solve_head_convergence_failure(self):
        """Test head solver with impossible parameters (maybe fails)."""
        # Zero conductivity everywhere except boundaries?
        K = np.zeros((4, 4), dtype=float)
        K[:, 0] = 1.0  # left boundary
        K[:, -1] = 1.0  # right boundary

        x = np.linspace(0, 3, 4)
        y = np.linspace(0, 3, 4)

        h, msg, is_successful = hydraulic.solve_head(K, x, y, dh=0.01, max_it=10)

        # May not converge due to zero conductivity interior
        # Just ensure function returns
        self.assertIsInstance(is_successful, bool)
        self.assertIsInstance(msg, str)


if __name__ == '__main__':
    unittest.main()