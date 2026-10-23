#!/usr/bin/env python3
"""Plot one plume and the concentration ensemble at a fixed spatial cell."""

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
import numpy as np


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="PNG path; defaults inside input-dir.")
    p.add_argument("--seed", type=int, default=0, help="Seed for selecting the realization.")
    p.add_argument("--cell-x", type=float, default=10.0, help="Fixed cell x coordinate in I_Y.")
    p.add_argument("--cell-y", type=float, default=10.0, help="Fixed cell y coordinate in I_Y.")
    return p


def analyze(args: argparse.Namespace) -> Path:
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    files = manifest["files"]
    if not files:
        raise ValueError("manifest contains no realizations")
    with np.load(args.input_dir / "grid.npz") as grid:
        x_faces = grid["x_faces"]
        y_faces = grid["y_faces"]
    ix = int(np.searchsorted(x_faces, args.cell_x, side="right") - 1)
    iy = int(np.searchsorted(y_faces, args.cell_y, side="right") - 1)
    shape = (len(y_faces) - 1, len(x_faces) - 1)
    if not (0 <= ix < shape[1] and 0 <= iy < shape[0]):
        raise ValueError("fixed cell coordinate must lie inside the domain")
    selected = int(np.random.default_rng(args.seed).integers(len(files)))
    plume = None
    ensemble = np.empty(len(files), dtype=float)
    for i, filename in enumerate(files):
        with np.load(args.input_dir / filename) as data:
            c = data["concentration"]
        if c.shape != shape or not np.all(np.isfinite(c)) or np.any(c < 0):
            raise ValueError(f"invalid concentration array in {filename}")
        ensemble[i] = c[iy, ix]
        if i == selected:
            plume = c
    assert plume is not None
    spatial_mean = float(np.mean(plume))
    spatial_variance = float(np.var(plume))
    ensemble_mean = float(np.mean(ensemble))
    ensemble_variance = float(np.var(ensemble))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    extent = [x_faces[0], x_faces[-1], y_faces[0], y_faces[-1]]
    image = axes[0].imshow(plume, origin="lower", extent=extent, aspect="equal",
                           cmap="viridis", vmin=0, vmax=max(float(np.max(plume)), 1e-12))
    axes[0].plot(args.cell_x, args.cell_y, "rx", markersize=7, markeredgewidth=1.5)
    axes[0].set_title(f"Realization {selected} at t*={manifest['t_star']:g}\n"
                      f"Spatial mean={spatial_mean:.3g}, variance={spatial_variance:.3g}")
    axes[0].set_xlabel("x / I_Y")
    axes[0].set_ylabel("y / I_Y")
    fig.colorbar(image, ax=axes[0], label="Relative concentration")

    bins = min(100, max(5, int(np.ceil(np.sqrt(len(ensemble))))))
    axes[1].hist(ensemble, bins=bins, edgecolor="white", color="#4477aa")
    axes[1].axvline(ensemble_mean, color="#bb3344", linestyle="--", linewidth=1.5,
                    label=f"Ensemble mean={ensemble_mean:.3g}")
    axes[1].set_title(f"Cell at ({args.cell_x:g}, {args.cell_y:g}) I_Y\n"
                      f"Ensemble variance={ensemble_variance:.3g}, n={len(ensemble)}")
    axes[1].set_xlabel("Relative concentration")
    axes[1].set_ylabel("Realizations")
    axes[1].legend()
    fig.suptitle("Spatial and ensemble concentration statistics (descriptive)", fontsize=13)
    output = args.output or (args.input_dir / "ergodicity_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}; selected realization={selected}; "
          f"spatial mean/variance={spatial_mean:.6g}/{spatial_variance:.6g}; "
          f"ensemble mean/variance={ensemble_mean:.6g}/{ensemble_variance:.6g}")
    return output


if __name__ == "__main__":
    analyze(parser().parse_args())
