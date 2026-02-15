"""
Load and display data from compare_par2_rk45_data.npz.

Usage:
  python show_npz.py [path_to.npz] [--plot]
"""

import argparse
from pathlib import Path

import numpy as np


def _summary(arr):
    """One-line summary for an array or scalar."""
    a = np.asarray(arr)
    if a.size == 0:
        return "empty"
    if a.ndim == 0 or a.size == 1:
        return f"{float(a):.6g}"
    return f"min={np.nanmin(a):.4g}, max={np.nanmax(a):.4g}, mean={np.nanmean(a):.4g}"


def main():
    default_npz = Path(__file__).parent / "compare_output" / "compare_par2_rk45_data.npz"
    parser = argparse.ArgumentParser(description="Show contents of compare PAR2/RK45 .npz file")
    parser.add_argument("npz_path", nargs="?", type=Path, default=default_npz, help="Path to .npz file")
    parser.add_argument("--plot", action="store_true", help="Plot paths on log-K and save figure")
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
        im = ax.imshow(log10k, origin="lower", cmap="viridis", aspect="equal")
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

    data.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
