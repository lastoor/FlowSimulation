#!/usr/bin/env python
"""
Test comparing MODFLOW (mf2005) and PETSc solutions for hydraulic head
and velocity fields.
"""

import sys
import os
import unittest
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


class TestCompareModflowPETSc(unittest.TestCase):
    """Test comparing MODFLOW and PETSc solutions."""

    @unittest.skipIf(not FLOPY_AVAILABLE, "flopy not available")
    def test_compare_modflow_petsc(self):
        """Compare MODFLOW and PETSc solutions for head and velocity."""
        # Test parameters
        num_pt = 100  # Grid size
        grad_head = 0.01  # Hydraulic head gradient
        
        # Generate logK field using fBm
        np.random.seed(42)  # For reproducibility
        fbm_gen = grf_generation.fBmGenerator(
            c=1.0, 
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
        
        # Run MODFLOW
        return_msg_mf = hydraulic.run_modflow(mf2005_config)
        print(return_msg_mf)
        self.assertIn('successfully', return_msg_mf.lower(), 
                     f"MODFLOW run failed: {return_msg_mf}")
        
        # Read MODFLOW head results
        model_ws = mf2005_config['model_ws']
        model_name = mf2005_config['model_name']
        
        # Open the head file
        headobj = flopy.utils.HeadFile(os.path.join(model_ws, f"{model_name}.hds"))
        
        # Get the head array for the first layer, first time step
        # MODFLOW shape: (nlay, nrow, ncol)
        head_mf = headobj.get_data(kstpkper=(0, 0))
        
        # Extract 2D head field (first layer)
        if head_mf.ndim == 3:
            head_mf_2d = head_mf[0, :, :]  # (nrow, ncol)
        else:
            head_mf_2d = head_mf
        
        # Read MODFLOW velocity field
        data = hydraulic.read_mt3d_file(str(path_wd / 'mf2005_output/mf2005_model.ftl'))
        QXX = data['QXX']  # MODFLOW velocity in x-direction
        QYY = data['QYY']  # MODFLOW velocity in y-direction
        
        # Run PETSc solver
        K = np.exp(logk_field)  # Convert logK to K
        x = np.arange(num_pt)  # Cell-center coordinates
        y = np.arange(num_pt)
        
        h_petsc, msg_petsc, is_success = hydraulic.solve_head(
            K, x, y, dh=grad_head, rtol=1e-6, atol=1e-20, max_it=10000
        )
        
        self.assertTrue(is_success, f"PETSc solver failed: {msg_petsc}")
        print(f"PETSc solver: {msg_petsc}")
        
        # Compute velocity from PETSc solution
        vx_petsc, vy_petsc = hydraulic.compute_velocity(K, x, y, h_petsc)
        
        # Ensure shapes match
        self.assertEqual(head_mf_2d.shape, h_petsc.shape, 
                        f"Head shape mismatch: MODFLOW {head_mf_2d.shape} vs PETSc {h_petsc.shape}")
        self.assertEqual(QXX.shape, vx_petsc.shape,
                        f"Velocity X shape mismatch: MODFLOW {QXX.shape} vs PETSc {vx_petsc.shape}")
        self.assertEqual(QYY.shape, vy_petsc.shape,
                        f"Velocity Y shape mismatch: MODFLOW {QYY.shape} vs PETSc {vy_petsc.shape}")
        
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
        
        # Compare velocity fields
        vx_diff = np.abs(QXX - vx_petsc)
        vy_diff = np.abs(QYY - vy_petsc)
        vx_rmse = np.sqrt(np.mean(vx_diff**2))
        vy_rmse = np.sqrt(np.mean(vy_diff**2))
        vx_max_diff = np.max(vx_diff)
        vy_max_diff = np.max(vy_diff)
        vx_mean_mf = np.mean(np.abs(QXX))
        vy_mean_mf = np.mean(np.abs(QYY))
        
        print(f"\nVelocity X comparison:")
        print(f"  RMSE: {vx_rmse:.6e}")
        print(f"  Max absolute difference: {vx_max_diff:.6e}")
        print(f"  Mean |MODFLOW QXX|: {vx_mean_mf:.6e}")
        print(f"  Relative RMSE: {vx_rmse/vx_mean_mf:.6e}")
        
        print(f"\nVelocity Y comparison:")
        print(f"  RMSE: {vy_rmse:.6e}")
        print(f"  Max absolute difference: {vy_max_diff:.6e}")
        print(f"  Mean |MODFLOW QYY|: {vy_mean_mf:.6e}")
        print(f"  Relative RMSE: {vy_rmse/vy_mean_mf:.6e}")
        
        # Assertions - allow for numerical differences
        # Head should be reasonably close (within 1% relative error)
        self.assertLess(head_rmse / head_mean_mf, 0.01, 
                        f"Head RMSE too large: {head_rmse/head_mean_mf:.6e}")
        
        # Velocity should be reasonably close (within 5% relative error, 
        # as velocity is more sensitive to discretization differences)
        if vx_mean_mf > 1e-10:  # Avoid division by zero
            self.assertLess(vx_rmse / vx_mean_mf, 0.05,
                           f"Velocity X RMSE too large: {vx_rmse/vx_mean_mf:.6e}")
        if vy_mean_mf > 1e-10:
            self.assertLess(vy_rmse / vy_mean_mf, 0.05,
                           f"Velocity Y RMSE too large: {vy_rmse/vy_mean_mf:.6e}")
        
        # Check that solutions are not identical (would indicate a bug)
        self.assertGreater(head_rmse, 1e-10, 
                          "Head solutions are too similar (possible bug)")


if __name__ == '__main__':
    unittest.main()
