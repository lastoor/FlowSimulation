"""
Load and display data from compare_par2_rk45_data.npz.

Usage:
  python show_npz.py [path_to.npz] [--plot]
  python show_npz.py [path_to.npz] --snapshots [DIR]   # visualize PAR2 snapshots (default DIR: par2_output/snapshots)
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
