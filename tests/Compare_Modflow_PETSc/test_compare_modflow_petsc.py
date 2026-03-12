#!/usr/bin/env python
"""
Test comparing MODFLOW (mf2005) and PETSc solutions for hydraulic head
and velocity fields.
"""

import argparse
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


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare MODFLOW (mf2005) and PETSc hydraulic head/velocity solutions.",
    )
    p.add_argument("--num-pt", type=int, default=100, help="Grid size in x/y.")
    p.add_argument("--dh", type=float, default=0.01, help="Hydraulic head gradient.")
    p.add_argument("--seed", type=int, default=56, help="Random seed for logK generation.")
    p.add_argument("--c", type=float, default=1.0, help="fBm scale parameter.")
    p.add_argument("--alpha", type=float, default=0.5, help="fBm alpha parameter.")
    p.add_argument("--minpadding", type=int, default=0, help="fBm padding.")
    p.add_argument("--rtol", type=float, default=1e-10, help="PETSc relative tolerance.")
    p.add_argument("--atol", type=float, default=1e-10, help="PETSc absolute tolerance.")
    p.add_argument("--max-it", type=int, default=10000, help="PETSc max iterations.")
    p.add_argument("--velocity-location", choices=("face", "cell"), default="face")
    p.add_argument("--align-with-modflow-velocity", dest="align_with_modflow", action="store_true", default=True)
    p.add_argument("--no-align-with-modflow-velocity", dest="align_with_modflow", action="store_false")
    p.add_argument("--spacing-mode", choices=("uniform", "center_faces"), default="center_faces")
    p.add_argument("--face-output", choices=("trimmed", "full"), default="trimmed")
    p.add_argument("--skip-figures", action="store_true", help="Do not save comparison figures.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.resolve(),
        help="Output directory for MODFLOW artifacts and figures.",
    )
    return p.parse_args()


def _default_opts():
    return argparse.Namespace(
        num_pt=100,
        dh=0.01,
        seed=56,
        c=10.0,
        alpha=0.5,
        minpadding=0,
        rtol=1e-10,
        atol=1e-10,
        max_it=10000,
        output_location="face",
        align_with_modflow=True,
        spacing_mode="uniform",
        face_output="trimmed",
        skip_figures=False,
        out_dir=Path(__file__).parent.resolve(),
    )


def _resolve_modflow_exe():
    """Resolve MODFLOW executable from env or common install path."""
    path_mf2005_env = os.environ.get('MODFLOW_EXE', None)
    if path_mf2005_env:
        return Path(path_mf2005_env).resolve()
    for p in [Path('~/bin/mf2005').expanduser()]:
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def run_compare_modflow_petsc(opts):
    """Run the MODFLOW vs PETSc comparison and return summary metrics."""
    if not FLOPY_AVAILABLE:
        raise RuntimeError("flopy not available")
    path_mf2005 = _resolve_modflow_exe()
    if path_mf2005 is None or not path_mf2005.exists():
        raise RuntimeError(
            "MODFLOW executable not found. Set MODFLOW_EXE environment variable or install in ~/bin/mf2005."
        )

    num_pt = int(opts.num_pt)
    x = np.arange(num_pt)
    y = np.arange(num_pt)
    grad_head = float(opts.dh)

    np.random.seed(int(opts.seed))
    fbm_gen = grf_generation.fBmGenerator(
        c=float(opts.c),
        alpha=float(opts.alpha),
        num_pt=num_pt,
        dim=2,
        minpadding=int(opts.minpadding),
    )
    field, msg = grf_generation.generate_logk_field(fbm_gen)
    if field is None:
        raise RuntimeError(f"Field generation failed: {msg}")
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got {field.ndim}D")
    logk_field = field

    path_wd = Path(opts.out_dir).resolve()
    path_wd.mkdir(parents=True, exist_ok=True)

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

    model_ws = mf2005_config['model_ws']
    model_name = mf2005_config['model_name']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return_msg_mf = hydraulic.run_modflow(mf2005_config)
    print(return_msg_mf)
    if 'successfully' not in return_msg_mf.lower():
        raise RuntimeError(f"MODFLOW run failed: {return_msg_mf}")

    headobj = flopy.utils.HeadFile(os.path.join(model_ws, f"{model_name}.hds"))
    try:
        head_mf = headobj.get_data(kstpkper=(0, 0))
        head_mf_2d = head_mf[0, :, :] if head_mf.ndim == 3 else head_mf
    finally:
        headobj.close()

    data = hydraulic.read_mt3d_file(str(path_wd / 'mf2005_output/mf2005_model.ftl'))
    QXX = data['QXX']
    QYY = data['QYY']

    K = np.exp(logk_field)
    h_petsc, msg_petsc, is_success = hydraulic.solve_head(
        K,
        x,
        y,
        dh=grad_head,
        rtol=float(opts.rtol),
        atol=float(opts.atol),
        max_it=int(opts.max_it),
        spacing_mode=opts.spacing_mode,
    )
    print(f"PETSc solver: {msg_petsc}")
    if not is_success:
        raise RuntimeError(f"PETSc solver failed: {msg_petsc}")

    vel_kw = {
        "output_location": opts.velocity_location,
        "align_with_modflow": bool(opts.align_with_modflow),
    }
    if opts.velocity_location == "face":
        vel_kw["face_output"] = opts.face_output
    vx_cmp, vy_cmp = hydraulic.compute_velocity(K, x, y, h_petsc, **vel_kw)
    vx_cell, vy_cell = hydraulic.compute_velocity(
        K, x, y, h_petsc, output_location='cell', align_with_modflow=bool(opts.align_with_modflow)
    )

    def overlap_2d(a, b):
        r = min(a.shape[0], b.shape[0])
        c = min(a.shape[1], b.shape[1])
        return (np.s_[:r, :c], np.s_[:r, :c])

    qxx_slice_v, vx_slice_v = overlap_2d(QXX, vx_cmp)
    qyy_slice_v, vy_slice_v = overlap_2d(QYY, vy_cmp)
    QXX_v = QXX[qxx_slice_v]
    QYY_v = QYY[qyy_slice_v]
    vx_v = vx_cmp[vx_slice_v]
    vy_v = vy_cmp[vy_slice_v]

    qxx_slice_c, vx_slice_c = overlap_2d(QXX, vx_cell)
    qyy_slice_c, vy_slice_c = overlap_2d(QYY, vy_cell)
    QXX_c = QXX[qxx_slice_c]
    QYY_c = QYY[qyy_slice_c]
    vx_cell_o = vx_cell[vx_slice_c]
    vy_cell_o = vy_cell[vy_slice_c]

    if head_mf_2d.shape != h_petsc.shape:
        raise RuntimeError(f"Head shape mismatch: MODFLOW {head_mf_2d.shape} vs PETSc {h_petsc.shape}")

    head_diff = np.abs(head_mf_2d - h_petsc)
    head_rmse = np.sqrt(np.mean(head_diff**2))
    head_max_diff = np.max(head_diff)
    head_mean_mf = np.mean(np.abs(head_mf_2d))
    print(f"\nHead comparison:")
    print(f"  RMSE: {head_rmse:.6e}")
    print(f"  Max absolute difference: {head_max_diff:.6e}")
    print(f"  Mean |MODFLOW head|: {head_mean_mf:.6e}")
    print(f"  Relative RMSE: {head_rmse/head_mean_mf:.6e}")

    def vel_stats(q, v):
        diff = np.abs(q - v)
        rmse = np.sqrt(np.mean(diff**2))
        mean_q = np.mean(np.abs(q))
        rel = rmse / mean_q if mean_q > 1e-10 else rmse
        if q.size <= 1:
            corr = 1.0
        else:
            corr = np.corrcoef(q.flatten(), v.flatten())[0, 1]
            corr = 1.0 if np.isnan(corr) else corr
        return rmse, mean_q, rel, corr

    vx_rmse_v, vx_mean_v, vx_rel_v, vx_corr_v = vel_stats(QXX_v, vx_v)
    vy_rmse_v, vy_mean_v, vy_rel_v, vy_corr_v = vel_stats(QYY_v, vy_v)
    print(
        f"\nVelocity comparison ({opts.velocity_location}, overlap shape {vx_v.shape} / {vy_v.shape}):"
    )
    print(f"  vx: RMSE={vx_rmse_v:.6e}, mean|QXX|={vx_mean_v:.6e}, rel={vx_rel_v:.6e}, corr={vx_corr_v:.4f}")
    print(f"  vy: RMSE={vy_rmse_v:.6e}, mean|QYY|={vy_mean_v:.6e}, rel={vy_rel_v:.6e}, corr={vy_corr_v:.4f}")

    vx_rmse_c, vx_mean_c, vx_rel_c, vx_corr_c = vel_stats(QXX_c, vx_cell_o)
    vy_rmse_c, vy_mean_c, vy_rel_c, vy_corr_c = vel_stats(QYY_c, vy_cell_o)
    print(f"\nVelocity comparison (cell-centered, overlap shape {vx_cell_o.shape} / {vy_cell_o.shape}):")
    print(f"  vx: RMSE={vx_rmse_c:.6e}, mean|QXX|={vx_mean_c:.6e}, rel={vx_rel_c:.6e}, corr={vx_corr_c:.4f}")
    print(f"  vy: RMSE={vy_rmse_c:.6e}, mean|QYY|={vy_mean_c:.6e}, rel={vy_rel_c:.6e}, corr={vy_corr_c:.4f}")

    if MATPLOTLIB_AVAILABLE and not opts.skip_figures:
        _save_comparison_figures(
            out_dir=path_wd,
            head_mf=head_mf_2d,
            head_petsc=h_petsc,
            QXX_c=QXX_c,
            QYY_c=QYY_c,
            vx_cell_o=vx_cell_o,
            vy_cell_o=vy_cell_o,
        )
    elif not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization (matplotlib not available).")

    return {
        "head_shape": head_mf_2d.shape,
        "head_rel": head_rmse / head_mean_mf if head_mean_mf > 1e-10 else head_rmse,
        "head_rmse": head_rmse,
        "velocity_rmse": (vx_rmse_v, vy_rmse_v),
        "velocity_corr": (vx_corr_v, vy_corr_v),
        "cell_velocity_rmse": (vx_rmse_c, vy_rmse_c),
        "cell_velocity_corr": (vx_corr_c, vy_corr_c),
    }


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
        if not FLOPY_AVAILABLE:
            self.skipTest("flopy not available")
        if _resolve_modflow_exe() is None:
            self.skipTest(
                "MODFLOW executable not found. Set MODFLOW_EXE environment variable or install in ~/bin/mf2005."
            )
        metrics = run_compare_modflow_petsc(_default_opts())
        self.assertEqual(metrics["head_shape"], (100, 100))
        self.assertLess(metrics["head_rel"], 0.05, f"Head RMSE too large: {metrics['head_rel']:.6e}")
        self.assertLess(metrics["velocity_rmse"][0], 0.1)
        self.assertLess(metrics["velocity_rmse"][1], 0.1)
        self.assertGreater(metrics["velocity_corr"][0], 0.5)
        self.assertGreater(metrics["velocity_corr"][1], 0.5)
        self.assertLess(metrics["cell_velocity_rmse"][0], 0.1)
        self.assertLess(metrics["cell_velocity_rmse"][1], 0.1)
        self.assertGreater(metrics["cell_velocity_corr"][0], 0.5)
        self.assertGreater(metrics["cell_velocity_corr"][1], 0.5)
        self.assertGreater(metrics["head_rmse"], 1e-10)


if __name__ == '__main__':
    run_compare_modflow_petsc(_parse_args())
