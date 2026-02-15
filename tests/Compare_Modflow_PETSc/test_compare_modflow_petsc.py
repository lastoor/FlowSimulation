#!/usr/bin/env python
"""
Test comparing MODFLOW (mf2005) and PETSc solutions for hydraulic head
and velocity fields.
"""

import sys
import os
import unittest
import warnings
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simulation import hydraulic
from simulation import grf_generation

# Check if flopy is available
try:
    import flopy
    FLOPY_AVAILABLE = True
except ImportError:
    FLOPY_AVAILABLE = False

# Check if matplotlib is available for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _save_comparison_figures(out_dir, head_mf, head_petsc, QXX_c, QYY_c, vx_cell_o, vy_cell_o):
    """Plot and save hydraulic head and velocity comparison (MODFLOW vs PETSc and difference)."""
    path_wd = Path(out_dir)
    path_wd.mkdir(parents=True, exist_ok=True)

    # --- Head: MODFLOW, PETSc, difference ---
    head_diff = np.abs(head_mf - head_petsc)
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in zip(
        axes1,
        [head_mf, head_petsc, head_diff],
        ['MODFLOW head', 'PETSc head', '|MODFLOW − PETSc|'],
    ):
        im = ax.imshow(data, origin='lower', aspect='equal')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    fig1.suptitle('Hydraulic head comparison')
    fig1.tight_layout()
    fig1.savefig(path_wd / 'head_comparison.png', dpi=150)
    plt.close(fig1)

    # --- Velocity: magnitude and difference (cell-centered overlap) ---
    mag_mf = np.sqrt(QXX_c**2 + QYY_c**2)
    mag_petsc = np.sqrt(vx_cell_o**2 + vy_cell_o**2)
    mag_diff = np.sqrt((QXX_c - vx_cell_o)**2 + (QYY_c - vy_cell_o)**2)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in zip(
        axes2,
        [mag_mf, mag_petsc, mag_diff],
        ['MODFLOW |v|', 'PETSc |v|', '|MODFLOW − PETSc| velocity'],
    ):
        im = ax.imshow(data, origin='lower', aspect='equal')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    fig2.suptitle('Velocity magnitude comparison')
    fig2.tight_layout()
    fig2.savefig(path_wd / 'velocity_magnitude_comparison.png', dpi=150)
    plt.close(fig2)

    # --- Velocity quiver (subsampled for clarity); arrow length proportional to magnitude ---
    nr, nc = vx_cell_o.shape
    step = max(1, min(nr, nc) // 15)
    jj = np.arange(0, nc, step)
    ii = np.arange(0, nr, step)
    J, I = np.meshgrid(jj, ii)

    U_mf = QXX_c[np.ix_(ii, jj)]
    V_mf = QYY_c[np.ix_(ii, jj)]
    U_petsc = vx_cell_o[np.ix_(ii, jj)]
    V_petsc = vy_cell_o[np.ix_(ii, jj)]
    dx_v = QXX_c - vx_cell_o
    dy_v = QYY_c - vy_cell_o
    U_diff = dx_v[np.ix_(ii, jj)]
    V_diff = dy_v[np.ix_(ii, jj)]

    mag_mf_sub = np.sqrt(U_mf**2 + V_mf**2)
    mag_petsc_sub = np.sqrt(U_petsc**2 + V_petsc**2)
    mag_diff_sub = np.sqrt(U_diff**2 + V_diff**2)
    max_mag = max(np.nanmax(mag_mf_sub), np.nanmax(mag_petsc_sub), np.nanmax(mag_diff_sub), 1e-12)
    desired_len = 0.15 * min(nr, nc)  # longest arrow ~15% of domain
    quiver_scale = max_mag / desired_len

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    axes3[0].quiver(J, I, U_mf, V_mf, scale=quiver_scale, scale_units='xy', angles='xy')
    axes3[0].set_title('MODFLOW velocity')
    axes3[0].set_aspect('equal')
    axes3[0].invert_yaxis()

    axes3[1].quiver(J, I, U_petsc, V_petsc, scale=quiver_scale, scale_units='xy', angles='xy')
    axes3[1].set_title('PETSc velocity')
    axes3[1].set_aspect('equal')
    axes3[1].invert_yaxis()

    axes3[2].quiver(J, I, U_diff, V_diff, scale=quiver_scale, scale_units='xy', angles='xy')
    axes3[2].set_title('Velocity difference (MODFLOW − PETSc)')
    axes3[2].set_aspect('equal')
    axes3[2].invert_yaxis()

    fig3.suptitle('Velocity field comparison')
    fig3.tight_layout()
    fig3.savefig(path_wd / 'velocity_quiver_comparison.png', dpi=150)
    plt.close(fig3)


class TestCompareModflowPETSc(unittest.TestCase):
    """Test comparing MODFLOW and PETSc solutions."""

    @unittest.skipIf(not FLOPY_AVAILABLE, "flopy not available")
    def test_compare_modflow_petsc(self):
        """Compare MODFLOW and PETSc solutions for head and velocity."""
        # Test parameters
        num_pt = 100  # Grid size
        x = np.arange(num_pt)
        y = np.arange(num_pt)
        grad_head = 0.01  # Hydraulic head gradient
        
        # Generate logK field using fBm
        np.random.seed(56)  # For reproducibility
        fbm_gen = grf_generation.fBmGenerator(
            c=10, 
            alpha=0.5, 
            num_pt=num_pt, 
            dim=2, 
            minpadding=0
        )
        field, msg = grf_generation.generate_logk_field(fbm_gen)
        self.assertIsNotNone(field, f"Field generation failed: {msg}")
        
        # Convert to 2D if needed (should already be 2D)
        if field.ndim == 2:
            logk_field = field
        else:
            raise ValueError(f"Expected 2D field, got {field.ndim}D")
        
        # Setup MODFLOW configuration
        # Use paths relative to test file for portability
        # Get project root directory (two levels up from test file)
        path_wd = Path(__file__).parent.resolve()
        path_wd.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        
        # MODFLOW executable: try common locations, or use environment variable
        # This allows users to override via environment variable
        path_mf2005_env = os.environ.get('MODFLOW_EXE', None)
        if path_mf2005_env:
            path_mf2005 = Path(path_mf2005_env).resolve()
        else:
            # Try common locations
            common_paths = [
                Path('~/bin/mf2005').expanduser(),
            ]
            path_mf2005 = None
            for p in common_paths:
                if p.exists() and p.is_file():
                    path_mf2005 = p.resolve()
                    break
        
        if path_mf2005 is None or not path_mf2005.exists():
            self.skipTest("MODFLOW executable not found. Set MODFLOW_EXE environment variable or install in common location.")
        mf2005_config = {
            'modflow_exe': str(path_mf2005),
            'model_ws': str(path_wd / 'mf2005_output'),
            'model_name': 'mf2005_model',
            'outftl_name': str(path_wd / 'mf2005_output/mf2005_model.ftl'),
            'grad_head': grad_head,
            'lx': num_pt,
            'ly': num_pt,
            'lz': 1,
            'dx': 1,
            'dy': 1,
            'dz': 1,
            'logk_field': logk_field,
        }
        
        # Run MODFLOW (suppress ResourceWarnings from flopy/subprocess)
        model_ws = mf2005_config['model_ws']
        model_name = mf2005_config['model_name']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return_msg_mf = hydraulic.run_modflow(mf2005_config)
        print(return_msg_mf)
        self.assertIn('successfully', return_msg_mf.lower(),
                     f"MODFLOW run failed: {return_msg_mf}")

        # Read MODFLOW head results; ensure head file is closed
        headobj = flopy.utils.HeadFile(os.path.join(model_ws, f"{model_name}.hds"))
        try:
            head_mf = headobj.get_data(kstpkper=(0, 0))
            if head_mf.ndim == 3:
                head_mf_2d = head_mf[0, :, :]  # (nrow, ncol)
            else:
                head_mf_2d = head_mf
        finally:
            headobj.close()

        # Read MODFLOW velocity field
        data = hydraulic.read_mt3d_file(str(path_wd / 'mf2005_output/mf2005_model.ftl'))
        QXX = data['QXX']  # MODFLOW velocity in x-direction
        QYY = data['QYY']  # MODFLOW velocity in y-direction
        
        # Run PETSc solver
        K = np.exp(logk_field)  # Convert logK to K

        h_petsc, msg_petsc, is_success = hydraulic.solve_head(
            K, x, y, dh=grad_head, rtol=1e-10, atol=1e-10, max_it=10000
        )
        
        self.assertTrue(is_success, f"PETSc solver failed: {msg_petsc}")
        print(f"PETSc solver: {msg_petsc}")
        
        # Compute velocity from PETSc (both face- and cell-centered)
        vx_face, vy_face = hydraulic.compute_velocity(
            K, x, y, h_petsc, output_location='face'
        )
        vx_cell, vy_cell = hydraulic.compute_velocity(
            K, x, y, h_petsc, output_location='cell'
        )
        
        def overlap_2d(a, b):
            """Return (a_slice, b_slice) so that a[a_slice].shape == b[b_slice].shape (overlapping region)."""
            r = min(a.shape[0], b.shape[0])
            c = min(a.shape[1], b.shape[1])
            return (np.s_[:r, :c], np.s_[:r, :c])
        
        # Compare both face and cell using overlapping regions when dimensions differ
        # --- Face-centered comparison (overlapping)
        qxx_slice_f, vx_slice_f = overlap_2d(QXX, vx_face)
        qyy_slice_f, vy_slice_f = overlap_2d(QYY, vy_face)
        QXX_f = QXX[qxx_slice_f]
        QYY_f = QYY[qyy_slice_f]
        vx_face_o = vx_face[vx_slice_f]
        vy_face_o = vy_face[vy_slice_f]
        
        # --- Cell-centered comparison (overlapping)
        qxx_slice_c, vx_slice_c = overlap_2d(QXX, vx_cell)
        qyy_slice_c, vy_slice_c = overlap_2d(QYY, vy_cell)
        QXX_c = QXX[qxx_slice_c]
        QYY_c = QYY[qyy_slice_c]
        vx_cell_o = vx_cell[vx_slice_c]
        vy_cell_o = vy_cell[vy_slice_c]
        
        # Ensure shapes match
        self.assertEqual(head_mf_2d.shape, h_petsc.shape, 
                        f"Head shape mismatch: MODFLOW {head_mf_2d.shape} vs PETSc {h_petsc.shape}")
        
        # Compare head fields
        head_diff = np.abs(head_mf_2d - h_petsc)
        head_rmse = np.sqrt(np.mean(head_diff**2))
        head_max_diff = np.max(head_diff)
        head_mean_mf = np.mean(np.abs(head_mf_2d))
        
        print(f"\nHead comparison:")
        print(f"  RMSE: {head_rmse:.6e}")
        print(f"  Max absolute difference: {head_max_diff:.6e}")
        print(f"  Mean |MODFLOW head|: {head_mean_mf:.6e}")
        print(f"  Relative RMSE: {head_rmse/head_mean_mf:.6e}")
        
        # Compare velocity fields: both face-centered and cell-centered (overlapping regions)
        def vel_stats(q, v):
            diff = np.abs(q - v)
            rmse = np.sqrt(np.mean(diff**2))
            mean_q = np.mean(np.abs(q))
            rel = rmse / mean_q if mean_q > 1e-10 else rmse
            if q.size <= 1:
                corr = 1.0
            else:
                corr = np.corrcoef(q.flatten(), v.flatten())[0, 1]
                corr = 1.0 if np.isnan(corr) else corr  # constant arrays
            return rmse, mean_q, rel, corr
        
        # Face-centered (overlapping)
        vx_rmse_f, vx_mean_f, vx_rel_f, vx_corr_f = vel_stats(QXX_f, vx_face_o)
        vy_rmse_f, vy_mean_f, vy_rel_f, vy_corr_f = vel_stats(QYY_f, vy_face_o)
        print(f"\nVelocity comparison (face-centered, overlap shape {vx_face_o.shape} / {vy_face_o.shape}):")
        print(f"  vx: RMSE={vx_rmse_f:.6e}, mean|QXX|={vx_mean_f:.6e}, rel={vx_rel_f:.6e}, corr={vx_corr_f:.4f}")
        print(f"  vy: RMSE={vy_rmse_f:.6e}, mean|QYY|={vy_mean_f:.6e}, rel={vy_rel_f:.6e}, corr={vy_corr_f:.4f}")
        
        # Cell-centered (overlapping)
        vx_rmse_c, vx_mean_c, vx_rel_c, vx_corr_c = vel_stats(QXX_c, vx_cell_o)
        vy_rmse_c, vy_mean_c, vy_rel_c, vy_corr_c = vel_stats(QYY_c, vy_cell_o)
        print(f"\nVelocity comparison (cell-centered, overlap shape {vx_cell_o.shape} / {vy_cell_o.shape}):")
        print(f"  vx: RMSE={vx_rmse_c:.6e}, mean|QXX|={vx_mean_c:.6e}, rel={vx_rel_c:.6e}, corr={vx_corr_c:.4f}")
        print(f"  vy: RMSE={vy_rmse_c:.6e}, mean|QYY|={vy_mean_c:.6e}, rel={vy_rel_c:.6e}, corr={vy_corr_c:.4f}")
        
        # Visualize and save head and velocity fields and their differences
        if MATPLOTLIB_AVAILABLE:
            _save_comparison_figures(
                out_dir=path_wd,
                head_mf=head_mf_2d,
                head_petsc=h_petsc,
                QXX_c=QXX_c,
                QYY_c=QYY_c,
                vx_cell_o=vx_face_o,
                vy_cell_o=vy_face_o,
            )
        else:
            print("Skipping visualization (matplotlib not available).")
        
        # Assertions - allow for numerical differences
        # Head: within 5% relative error (different solvers/discretizations)
        head_rel = head_rmse / head_mean_mf if head_mean_mf > 1e-10 else head_rmse
        self.assertLess(head_rel, 0.05, 
                        f"Head RMSE too large: {head_rel:.6e} (threshold: 0.05)")
        
        # Velocity: require both face and cell comparisons to pass (absolute RMSE and correlation)
        self.assertLess(vx_rmse_f, 0.1, f"Velocity X (face) absolute RMSE too large: {vx_rmse_f:.6e}")
        self.assertLess(vy_rmse_f, 0.1, f"Velocity Y (face) absolute RMSE too large: {vy_rmse_f:.6e}")
        self.assertGreater(vx_corr_f, 0.5, f"Velocity X (face) correlation too low: {vx_corr_f:.6f}")
        self.assertGreater(vy_corr_f, 0.5, f"Velocity Y (face) correlation too low: {vy_corr_f:.6f}")
        self.assertLess(vx_rmse_c, 0.1, f"Velocity X (cell) absolute RMSE too large: {vx_rmse_c:.6e}")
        self.assertLess(vy_rmse_c, 0.1, f"Velocity Y (cell) absolute RMSE too large: {vy_rmse_c:.6e}")
        self.assertGreater(vx_corr_c, 0.5, f"Velocity X (cell) correlation too low: {vx_corr_c:.6f}")
        self.assertGreater(vy_corr_c, 0.5, f"Velocity Y (cell) correlation too low: {vy_corr_c:.6f}")
        
        # Check that solutions are not identical (would indicate a bug)
        self.assertGreater(head_rmse, 1e-10, 
                          "Head solutions are too similar (possible bug)")


if __name__ == '__main__':
    unittest.main()
