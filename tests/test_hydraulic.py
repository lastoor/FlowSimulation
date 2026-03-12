#!/usr/bin/env python
"""
Unit tests for hydraulic.py
"""

import sys
import os
import unittest
import numpy as np
import tempfile
from pathlib import Path

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

    def test_get_spacing_and_faces_nonuniform(self):
        centers = np.array([1.0, 2.0, 4.0], dtype=float)
        d_cell, d_face = hydraulic.get_spacing_and_faces(centers)
        np.testing.assert_array_almost_equal(d_cell, np.array([1.0, 1.5, 2.0]))
        np.testing.assert_array_almost_equal(d_face, np.array([1.0, 1.25, 1.75, 2.0]))

    def test_mt3d_read_write_roundtrip(self):
        """write_mt3d_file output is parseable by read_mt3d_file."""
        variables = {
            "header": " 'MT3D4.00.00'           0           0           0           0           0",
            "THKSAT": np.array([[1.0, 2.5, -3.0], [4.0, 5.0, 6.0]], dtype=float),
            "QXX": np.array([[0.0, 1e-3, -2e-3], [3.0, -4.0, 5.5]], dtype=float),
            "QYY": np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=float),
            "CNH": np.array(
                [
                    [1, 1, 1, 1.25e-2],
                    [1, 3, 1, -2.5],
                ],
                dtype=float,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path_ftl = Path(tmpdir) / "roundtrip.ftl"
            hydraulic.write_mt3d_file(str(path_ftl), variables)
            parsed = hydraulic.read_mt3d_file(str(path_ftl))

        self.assertEqual(parsed["header"], variables["header"])
        np.testing.assert_allclose(parsed["THKSAT"], variables["THKSAT"], rtol=0, atol=0)
        np.testing.assert_allclose(parsed["QXX"], variables["QXX"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(parsed["QYY"], variables["QYY"], rtol=1e-12, atol=0)

        # CNH parsed dtype may be float due to mixed columns; compare columns explicitly.
        self.assertEqual(parsed["CNH"].shape[0], variables["CNH"].shape[0])
        np.testing.assert_array_equal(parsed["CNH"][:, :3].astype(int), variables["CNH"][:, :3].astype(int))
        np.testing.assert_allclose(parsed["CNH"][:, 3], variables["CNH"][:, 3], rtol=1e-12, atol=0)

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

    def test_compute_velocity_face_alignment_default_matches_modflow(self):
        """Default trimmed face output should zero the outermost right/top faces."""
        nx, ny = 5, 5
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 4, ny)
        X, Y = np.meshgrid(x, y)
        h = X + 2.0 * Y

        qxx, qyy = hydraulic.compute_velocity(K, x, y, h, output_location="face")

        np.testing.assert_array_almost_equal(qxx[:, -1], 0.0)
        np.testing.assert_array_almost_equal(qyy[-1, :], 0.0)

    def test_compute_velocity_face_alignment_false_keeps_boundary_flux_trimmed(self):
        """Non-aligned trimmed face output should preserve computed outer-edge fluxes."""
        nx, ny = 5, 5
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 4, ny)
        X, Y = np.meshgrid(x, y)
        h = X + 2.0 * Y

        qxx, qyy = hydraulic.compute_velocity(
            K, x, y, h, output_location="face", align_with_modflow=False
        )

        self.assertTrue(np.all(np.isfinite(qxx)))
        self.assertTrue(np.all(np.isfinite(qyy)))
        np.testing.assert_array_almost_equal(qxx[:, -1], -1.0)
        np.testing.assert_array_almost_equal(qyy[-1, :], -2.0)

    def test_compute_velocity_face_output_full_shapes(self):
        """Full face output should return the staggered-grid face arrays."""
        nx, ny = 5, 4
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 3, ny)
        X, Y = np.meshgrid(x, y)
        h = X + 2.0 * Y

        vx_face, vy_face = hydraulic.compute_velocity(
            K, x, y, h, output_location="face", face_output="full", align_with_modflow=False
        )

        self.assertEqual(vx_face.shape, (ny, nx + 1))
        self.assertEqual(vy_face.shape, (ny + 1, nx))
        np.testing.assert_array_almost_equal(vx_face[:, -1], -1.0)
        np.testing.assert_array_almost_equal(vy_face[-1, :], -2.0)

    def test_compute_velocity_face_output_full_alignment_matches_modflow(self):
        """Full face output should also zero the outer right/top faces when aligned."""
        nx, ny = 5, 5
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 4, ny)
        X, Y = np.meshgrid(x, y)
        h = X + 2.0 * Y

        vx_face, vy_face = hydraulic.compute_velocity(
            K, x, y, h, output_location="face", face_output="full", align_with_modflow=True
        )

        np.testing.assert_array_almost_equal(vx_face[:, -1], 0.0)
        np.testing.assert_array_almost_equal(vy_face[-1, :], 0.0)

    def test_compute_velocity_cell_uses_full_face_average(self):
        """Cell-centered output should average the full left/right and bottom/top faces."""
        nx, ny = 5, 5
        K = np.ones((ny, nx), dtype=float)
        x = np.linspace(0, 4, nx)
        y = np.linspace(0, 4, ny)
        X, Y = np.meshgrid(x, y)
        h = X + 2.0 * Y

        vx_aligned, vy_aligned = hydraulic.compute_velocity(
            K, x, y, h, output_location="cell", align_with_modflow=True
        )
        vx_free, vy_free = hydraulic.compute_velocity(
            K, x, y, h, output_location="cell", align_with_modflow=False
        )

        np.testing.assert_array_almost_equal(vx_aligned[:, 0], -1.0)
        np.testing.assert_array_almost_equal(vy_aligned[0, :], -2.0)
        np.testing.assert_array_almost_equal(vx_aligned[:, -1], -0.5)
        np.testing.assert_array_almost_equal(vy_aligned[-1, :], -1.0)
        np.testing.assert_array_almost_equal(vx_free[:, 0], -1.0)
        np.testing.assert_array_almost_equal(vy_free[0, :], -2.0)
        np.testing.assert_array_almost_equal(vx_free[:, -1], -1.0)
        np.testing.assert_array_almost_equal(vy_free[-1, :], -2.0)

    @unittest.skipIf(not PETSC_AVAILABLE, "petsc4py not available")
    def test_solve_head_invalid_spacing_mode(self):
        K = np.ones((3, 3), dtype=float)
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 2, 3)
        with self.assertRaises(ValueError):
            hydraulic.solve_head(K, x, y, spacing_mode="bad_mode")

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
    def test_solve_head_center_faces_nonuniform_grid(self):
        """Center-face spacing mode should solve on nonuniform cell-center grids."""
        K = np.ones((4, 4), dtype=float)
        x = np.array([0.5, 1.5, 3.5, 6.5], dtype=float)
        y = np.array([1.0, 2.5, 4.5, 7.5], dtype=float)

        h, msg, is_successful = hydraulic.solve_head(
            K, x, y, dh=0.01, spacing_mode="center_faces"
        )

        self.assertTrue(is_successful, msg=f"Solver failed: {msg}")
        self.assertEqual(h.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(h)))
        for row in range(4):
            for col in range(1, 4):
                self.assertLess(h[row, col], h[row, col - 1] + 1e-10)

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
