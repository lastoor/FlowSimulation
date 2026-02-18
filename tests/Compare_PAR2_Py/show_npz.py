"""
Load and display data from compare_par2_rk45_data.npz.

Usage:
  python show_npz.py [path_to.npz] [--plot]
  python show_npz.py [path_to.npz] --snapshots [DIR]   # visualize PAR2 snapshots (default DIR: par2_output/snapshots)
  python show_npz.py [path_to.npz] --accuracy         # path-velocity alignment (angle vs local velocity)
"""

import argparse
import re
from pathlib import Path
import cmcrameri

import numpy as np


def _summary(arr):
    """One-line summary for an array or scalar."""
    a = np.asarray(arr)
    if a.size == 0:
        return "empty"
    if a.ndim == 0 or a.size == 1:
        return f"{float(a):.6g}"
    return f"min={np.nanmin(a):.4g}, max={np.nanmax(a):.4g}, mean={np.nanmean(a):.4g}"


def _load_snapshot_csv(path):
    """Load particle x,y from a PAR2 snapshot CSV. Returns (x, y) arrays or (None, None) if empty."""
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2))
    except Exception:
        return None, None
    if data.size == 0 or data.ndim != 2:
        return None, None
    return data[:, 0], data[:, 1]


def _load_snapshots(snapshot_dir):
    """Glob snap_*.csv in snapshot_dir, sort by step number, return list of (step, x, y)."""
    snapshot_dir = Path(snapshot_dir).resolve()
    if not snapshot_dir.is_dir():
        return None, []
    files = sorted(snapshot_dir.glob("snap_*.csv"), key=lambda p: _snap_step(p.name))
    result = []
    for path in files:
        x, y = _load_snapshot_csv(path)
        if x is not None and x.size > 0:
            step = _snap_step(path.name)
            result.append((step, x, y))
    return snapshot_dir, result


def _snap_step(name):
    """Extract numeric suffix from snap_123.csv -> 123."""
    m = re.match(r"snap_(\d+)\.csv", name)
    return int(m.group(1)) if m else 0


def _bilinear_sample_2d(grid, x, y, ny, nx):
    """Sample 2D grid at (x, y) with bilinear interpolation. x, y are 1D arrays in [0, nx-1], [0, ny-1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, nx - 1.0 - 1e-10)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, ny - 1.0 - 1e-10)
    i0 = np.floor(x).astype(np.intp)
    j0 = np.floor(y).astype(np.intp)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    tx = x - i0
    ty = y - j0
    v00 = grid[j0, i0]
    v10 = grid[j0, i1]
    v01 = grid[j1, i0]
    v11 = grid[j1, i1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def _path_velocity_accuracy(vx, vy, x_path, y_path, eps_speed=1e-30):
    """
    Compute angle between path tangent and local velocity at each point.

    Returns mean absolute angle (degrees), RMS angle (degrees), and number of valid points.
    Skips points where tangent or velocity is zero.
    """
    x_path = np.asarray(x_path, dtype=np.float64).ravel()
    y_path = np.asarray(y_path, dtype=np.float64).ravel()
    n = len(x_path)
    if n < 2:
        return np.nan, np.nan, 0

    ny, nx = vx.shape
    if vy.shape != (ny, nx):
        return np.nan, np.nan, 0

    # Tangents: central difference interior, forward/backward at endpoints
    dx = np.empty(n)
    dy = np.empty(n)
    dx[0] = x_path[1] - x_path[0]
    dy[0] = y_path[1] - y_path[0]
    if n > 2:
        dx[1:-1] = x_path[2:] - x_path[:-2]
        dy[1:-1] = y_path[2:] - y_path[:-2]
    dx[-1] = x_path[-1] - x_path[-2]
    dy[-1] = y_path[-1] - y_path[-2]

    t_norm = np.sqrt(dx * dx + dy * dy)
    valid_t = t_norm > eps_speed
    if not np.any(valid_t):
        return np.nan, np.nan, 0

    # Interpolate velocity at path points
    vx_at = _bilinear_sample_2d(vx, x_path, y_path, ny, nx)
    vy_at = _bilinear_sample_2d(vy, x_path, y_path, ny, nx)
    v_norm = np.sqrt(vx_at * vx_at + vy_at * vy_at)
    valid_v = v_norm > eps_speed
    valid = valid_t & valid_v
    if not np.any(valid):
        return np.nan, np.nan, 0

    tx = np.where(valid_t, dx / t_norm, 0.0)
    ty = np.where(valid_t, dy / t_norm, 0.0)
    cos_theta = np.where(valid, (tx * vx_at + ty * vy_at) / (v_norm + 1e-300), np.nan)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    theta_deg_valid = theta_deg[valid]
    n_valid = int(np.sum(valid))
    mean_angle_deg = float(np.mean(np.abs(theta_deg_valid)))
    rms_angle_deg = float(np.sqrt(np.mean(theta_deg_valid ** 2)))
    return mean_angle_deg, rms_angle_deg, n_valid


def main():
    script_dir = Path(__file__).parent.resolve()
    default_npz = script_dir / "compare_output" / "compare_par2_rk45_data.npz"
    default_snapshots = script_dir / "par2_output" / "snapshots"
    parser = argparse.ArgumentParser(description="Show contents of compare PAR2/RK45 .npz file")
    parser.add_argument("npz_path", nargs="?", type=Path, default=default_npz, help="Path to .npz file")
    parser.add_argument("--plot", action="store_true", help="Plot paths on log-K and save figure")
    parser.add_argument(
        "--snapshots",
        nargs="?",
        const=True,
        default=None,
        help="Visualize PAR2 snapshots from DIR (default: par2_output/snapshots)",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Report path-velocity alignment (angle between path tangent and local velocity) for each path.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        return 1

    data = np.load(npz_path, allow_pickle=False)
    print(f"Loaded: {npz_path}\nKeys: {list(data.keys())}\n")

    for key in sorted(data.keys()):
        arr = data[key]
        if arr.size == 0:
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype} (empty)")
        elif arr.ndim == 0 or arr.size == 1:
            print(f"  {key}: {_summary(arr)}")
        else:
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, {_summary(arr)}")

    if args.accuracy:
        if "vx" not in data or "vy" not in data:
            print("\nSkipping path-velocity accuracy (no vx/vy in npz).")
        else:
            vx = data["vx"]
            vy = data["vy"]
            if vx.ndim != 2 or vy.shape != vx.shape:
                print("\nSkipping path-velocity accuracy (vx/vy not 2D or shape mismatch).")
            else:
                print("\nPath–velocity alignment (angle between path tangent and local velocity):")
                path_names = [
                    ("PAR2", "par2_x", "par2_y"),
                    ("RK45 CPU", "rk45_cpu_x", "rk45_cpu_y"),
                    ("RK45 GPU", "rk45_gpu_x", "rk45_gpu_y"),
                    ("RK4 FP", "rk4_fp_x", "rk4_fp_y"),
                ]
                for label, key_x, key_y in path_names:
                    if key_x not in data or key_y not in data:
                        continue
                    x_arr = data[key_x]
                    y_arr = data[key_y]
                    if x_arr.size < 2 or y_arr.size < 2:
                        continue
                    mean_deg, rms_deg, n_valid = _path_velocity_accuracy(vx, vy, x_arr, y_arr)
                    if n_valid == 0:
                        print(f"  {label:10s}: (no valid points)")
                    else:
                        print(f"  {label:10s}: mean |θ| = {mean_deg:.2f}°, RMS θ = {rms_deg:.2f}°  (n={n_valid} points)")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("--plot requires matplotlib. Skipping plot.")
            return 0

        logk = data["logk_field"]
        log10k = logk / np.log(10.0)
        ny, nx = log10k.shape

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(log10k, origin="lower", cmap=cmcrameri.cm.batlow, aspect="equal")
        plt.colorbar(im, ax=ax, label="log10 K", fraction=0.046, pad=0.04)

        t_cpu = data.get("rk45_cpu_time", np.nan)
        if np.isfinite(t_cpu) and "rk45_cpu_x" in data and data["rk45_cpu_x"].size > 0:
            ax.plot(data["rk45_cpu_x"], data["rk45_cpu_y"], "w--", lw=2, label=f"RK45 CPU (t={t_cpu:.3g})")

        t_gpu = data.get("rk45_gpu_time", np.nan)
        if np.isfinite(t_gpu) and "rk45_gpu_x" in data and data["rk45_gpu_x"].size > 0:
            ax.plot(data["rk45_gpu_x"], data["rk45_gpu_y"], "c-.", lw=2, label=f"RK45 GPU (t={t_gpu:.3g})")

        t_p2 = data.get("par2_time", np.nan)
        if np.isfinite(t_p2) and "par2_x" in data and data["par2_x"].size > 0:
            ax.plot(data["par2_x"], data["par2_y"], color="#D62728", lw=2, label=f"PAR2 (t={t_p2:.3g})")

        t_rk4 = data.get("rk4_fp_time", np.nan)
        if np.isfinite(t_rk4) and "rk4_fp_x" in data and data["rk4_fp_x"].size > 0:
            ax.plot(data["rk4_fp_x"], data["rk4_fp_y"], "m-", lw=2, label=f"RK4 FP (t={t_rk4:.3g})")

        ax.set_title("Paths from .npz")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = npz_path.parent / "show_npz_plot.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved plot: {out_path}")

    if args.snapshots is not None:
        snap_dir = Path(args.snapshots).resolve() if args.snapshots is not True else default_snapshots
        _, snap_list = _load_snapshots(snap_dir)
        if not snap_list:
            if not snap_dir.exists():
                print(f"\nSnapshot dir not found: {snap_dir}. Skipping snapshot visualization.")
            else:
                print(f"\nNo snap_*.csv files (or all empty) in {snap_dir}. Skipping snapshot visualization.")
        else:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib import cm
            except ImportError:
                print("\n--snapshots requires matplotlib. Skipping snapshot visualization.")
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                n_snap = len(snap_list)
                cmap = cmcrameri.cm.batlow.resampled(n_snap+1)
                for idx, (step, x, y) in enumerate(snap_list):
                    ax.scatter(x, y, s=10, alpha=1, c=cmap(idx))
                # Overlay fastest paths from npz on top
                t_cpu = data.get("rk45_cpu_time", np.nan)
                if np.isfinite(t_cpu) and "rk45_cpu_x" in data and data["rk45_cpu_x"].size > 0:
                    ax.plot(data["rk45_cpu_x"], data["rk45_cpu_y"], "w--", lw=2.5, label=f"RK45 CPU (t={t_cpu:.3g})")
                t_gpu = data.get("rk45_gpu_time", np.nan)
                if np.isfinite(t_gpu) and "rk45_gpu_x" in data and data["rk45_gpu_x"].size > 0:
                    ax.plot(data["rk45_gpu_x"], data["rk45_gpu_y"], "c-.", lw=2.5, label=f"RK45 GPU (t={t_gpu:.3g})")
                t_p2 = data.get("par2_time", np.nan)
                if np.isfinite(t_p2) and "par2_x" in data and data["par2_x"].size > 0:
                    ax.plot(data["par2_x"], data["par2_y"], color="#D62728", lw=2.5, label=f"PAR2 (t={t_p2:.3g})")
                t_rk4 = data.get("rk4_fp_time", np.nan)
                if np.isfinite(t_rk4) and "rk4_fp_x" in data and data["rk4_fp_x"].size > 0:
                    ax.plot(data["rk4_fp_x"], data["rk4_fp_y"], "m-", lw=2.5, label=f"RK4 FP (t={t_rk4:.3g})")
                all_x = np.concatenate([s[1] for s in snap_list])
                all_y = np.concatenate([s[2] for s in snap_list])
                ax.set_xlim(0, data["logk_field"].shape[0])
                ax.set_ylim(0, data["logk_field"].shape[1])
                ax.set_aspect("equal")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title("PAR2 snapshots (colored by time) and fastest paths")
                ax.legend(loc="best")
                fig.tight_layout()
                out_snap = npz_path.parent / "show_npz_snapshots.png"
                fig.savefig(out_snap, dpi=150)
                plt.close(fig)
                print(f"\nSaved snapshot plot: {out_snap} ({n_snap} snapshots)")

    data.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
