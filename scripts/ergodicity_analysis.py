#!/usr/bin/env python3
"""Create four PDF figures from saved concentration realizations."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flowsimulation-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "flowsimulation-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None, help="PDF directory; defaults to input-dir.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell-x", type=float, default=10.0)
    p.add_argument("--cell-y", type=float, default=10.0)
    p.add_argument("--window-size", type=int, default=1, help="Odd cells per side: 1, 3, 5, ...")
    p.add_argument("--hist-bins", type=int, default=None,
                   help="Default: min(100, max(5, ceil(sqrt(n_realizations)))).")
    return p


def region_patches(source, sample):
    sx0, sx1, sy0, sy1 = source
    x0, x1, y0, y1 = sample
    return (
        Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                  facecolor="#FFFFFF", edgecolor="#FFFFFF", alpha=0.6, linewidth=0, label="Source region"),
        Rectangle((x0, y0), x1 - x0, y1 - y0,
                  facecolor="#FC1D1D", edgecolor="#FC1D1D", linewidth=2, label="Sample region"),
    )


def save_map(field, x_faces, y_faces, title, label, path, source, sample):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    image = ax.imshow(field, origin="lower",
                      extent=[x_faces[0], x_faces[-1], y_faces[0], y_faces[-1]],
                      aspect="equal", cmap="viridis", vmin=0,
                      vmax=max(float(np.max(field)), 1e-12))
    for patch in region_patches(source, sample):
        ax.add_patch(patch)
    ax.set(title=title, xlabel="$x / I_Y$", ylabel="$y / I_Y$")
    ax.legend(loc="best", facecolor="white", framealpha=0.2, labelcolor="white")
    colorbar_ax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.12)
    fig.colorbar(image, cax=colorbar_ax, label=label)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> list[Path]:
    if args.window_size < 1 or args.window_size % 2 != 1:
        raise ValueError("window-size must be a positive odd integer")
    if args.hist_bins is not None and args.hist_bins < 1:
        raise ValueError("hist-bins must be a positive integer")
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    files = manifest["files"]
    if not files:
        raise ValueError("manifest contains no realizations")
    with np.load(args.input_dir / "grid.npz") as grid:
        x_faces, y_faces = grid["x_faces"], grid["y_faces"]
    ix = int(np.searchsorted(x_faces, args.cell_x, side="right") - 1)
    iy = int(np.searchsorted(y_faces, args.cell_y, side="right") - 1)
    shape = (len(y_faces) - 1, len(x_faces) - 1)
    radius = args.window_size // 2
    x0, x1, y0, y1 = ix - radius, ix + radius + 1, iy - radius, iy + radius + 1
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("sample window must lie entirely inside the domain")
    source = tuple(manifest["source_bounds"])
    sample = (x_faces[x0], x_faces[x1], y_faces[y0], y_faces[y1])

    selected = int(np.random.default_rng(args.seed).integers(len(files)))
    plume = None
    samples = np.empty(len(files), dtype=float)
    mean = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)
    for count, filename in enumerate(files, start=1):
        with np.load(args.input_dir / filename) as data:
            c = data["concentration"].astype(np.float64)
        if c.shape != shape or not np.all(np.isfinite(c)) or np.any(c < 0):
            raise ValueError(f"invalid concentration array in {filename}")
        samples[count - 1] = np.mean(c[y0:y1, x0:x1])
        if count - 1 == selected:
            plume = c.copy()
        delta = c - mean
        mean += delta / count
        m2 += delta * (c - mean)
    assert plume is not None
    variance = np.maximum(m2 / len(files), 0.0)
    spatial_mean, spatial_variance = float(np.mean(plume)), float(np.var(plume))
    sample_mean, sample_variance = float(np.mean(samples)), float(np.var(samples))
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in (
        "sample_realization.pdf", "sample_histogram.pdf",
        "ensemble_mean.pdf", "ensemble_variance.pdf")]
    save_map(plume, x_faces, y_faces,
             f"Realization {selected} at t={manifest['t_star']:g} days\n"
             f"Spatial mean={spatial_mean:.3g}, spatial variance={spatial_variance:.3g}",
             "$C/C_0$", paths[0], source, sample)

    bins = args.hist_bins if args.hist_bins is not None else min(100, max(5, int(np.ceil(np.sqrt(len(files))))))
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    ax.hist(samples, bins=bins, edgecolor="white", color="#4477aa")
    ax.axvline(sample_mean, color="#bb3344", linestyle="--", linewidth=1.5,
               label=r"$\mathbb{E}[C/C_0]$" + f"={sample_mean:.3g}")
    ax.set(title=f"{args.window_size} x {args.window_size} cells around "
                 f"({args.cell_x:g}, {args.cell_y:g}) $I_Y$\n"
                 r"$\mathrm{Var}[C/C_0]$" + f"={sample_variance:.3g}, n={len(files)}",
           xlabel="$C/C_0$", ylabel="Realizations")
    ax.legend(loc="best", facecolor="white", framealpha=0.3)
    fig.savefig(paths[1])
    plt.close(fig)
    save_map(mean, x_faces, y_faces, f"Ensemble mean concentration at t={manifest['t_star']:g} days",
             "$\mathbb{E}[C/C_0]$", paths[2], source, sample)
    save_map(variance, x_faces, y_faces,
             f"Ensemble concentration variance at t={manifest['t_star']:g} days",
             "$\mathrm{Var}[C/C_0]$", paths[3], source, sample)
    for path in paths:
        print(f"Saved {path}")
    print(f"Selected realization={selected}; spatial mean/variance={spatial_mean:.6g}/{spatial_variance:.6g}; "
          f"sample ensemble mean/variance={sample_mean:.6g}/{sample_variance:.6g}")
    return paths


if __name__ == "__main__":
    analyze(parser().parse_args())
