#!/usr/bin/env python3
"""Monte Carlo concentration snapshots from ergodic lognormal K fields."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import grf_generation, hydraulic, particle_tracking


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def nonnegative(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ly", type=positive, default=20.0, help="Domain height in I_Y; L_x=20 I_Y.")
    p.add_argument("--logk-variance", type=nonnegative, default=2.0)
    p.add_argument("--dx", type=positive, default=0.25, help="Square cell size in I_Y.")
    p.add_argument("--num-realizations", type=int, default=20)
    p.add_argument("--num-particles", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-star", type=positive, default=20.0)
    p.add_argument("--source-width", type=positive, default=1.5, help="Source x size in I_Y.")
    p.add_argument("--source-height", type=positive, default=6.0, help="Source y size in I_Y.")
    p.add_argument("--dt", type=positive, default=0.05)
    p.add_argument("--steps", type=int, default=None,
                   help="PyPar2 step limit; defaults to ceil(t-star/dt).")
    p.add_argument("--head-gradient", type=positive, default=0.1)
    p.add_argument("--porosity", type=positive, default=1.0)
    p.add_argument("--molecular-diffusion", type=nonnegative, default=1e-9)
    p.add_argument("--alpha-l", type=nonnegative, default=0.1)
    p.add_argument("--alpha-t", type=nonnegative, default=0.01)
    p.add_argument("--taichi-arch", choices=("cpu", "gpu"), default="gpu")
    return p


def grid_count(length: float, dx: float, name: str) -> int:
    n = round(length / dx)
    if n < 2 or not np.isclose(n * dx, length, rtol=0, atol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of dx with at least two cells")
    return n


def geometry(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    lx = 20.0
    nx = grid_count(lx, args.dx, "L_x")
    ny = grid_count(args.ly, args.dx, "L_y")
    x0, x1 = 5.0, 5.0 + args.source_width
    y0, y1 = (args.ly - args.source_height) / 2, (args.ly + args.source_height) / 2
    if min(x0, lx - x1, y0, args.ly - y1) < 5.0 - 1e-10:
        raise ValueError("source must remain at least 5 I_Y from all four boundaries")
    x_faces = np.linspace(0.0, lx, nx + 1)
    y_faces = np.linspace(0.0, args.ly, ny + 1)
    return x_faces, y_faces, (x0, x1, y0, y1)


def concentration(snapshots: dict, x_faces: np.ndarray, y_faces: np.ndarray,
                  source: tuple[float, float, float, float], n_particles: int) -> tuple[np.ndarray, int]:
    sx = np.asarray(snapshots["x"])[:, 0]
    sy = np.asarray(snapshots["y"])[:, 0]
    valid = np.isfinite(sx) & np.isfinite(sy)
    valid &= (sx >= x_faces[0]) & (sx < x_faces[-1])
    valid &= (sy >= y_faces[0]) & (sy < y_faces[-1])
    counts, _, _ = np.histogram2d(sy[valid], sx[valid], bins=(y_faces, x_faces))
    source_area = (source[1] - source[0]) * (source[3] - source[2])
    cell_area = (x_faces[1] - x_faces[0]) * (y_faces[1] - y_faces[0])
    return (counts * source_area / (n_particles * cell_area)).astype(np.float32), int(valid.sum())


def run(args: argparse.Namespace) -> None:
    if args.num_realizations < 1 or args.num_particles < 1:
        raise ValueError("num-realizations and num-particles must be positive integers")
    x_faces, y_faces, source = geometry(args)
    # PyPar2 advances elapsed time in float32; one or two extra steps avoid
    # losing a snapshot at an exact multiple of dt to roundoff.
    steps = math.ceil(args.t_star / args.dt) + 2 if args.steps is None else args.steps
    if steps < 2 or args.t_star > (steps - 1) * args.dt + 1e-9:
        raise ValueError("t-star needs at least one step of margin before the step limit")
    x = (x_faces[:-1] + x_faces[1:]) / 2
    y = (y_faces[:-1] + y_faces[1:]) / 2
    cov = grf_generation.ExpCov(sigma=math.sqrt(args.logk_variance), length_scale=1.0)
    grf = grf_generation.GaussianRandomFieldCircEmbed(
        np.zeros((len(y), len(x))), cov, (y, x)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index in range(args.num_realizations):
        field_seed = int(np.random.SeedSequence([args.seed, index, 0]).generate_state(1)[0])
        particle_seed = int(np.random.SeedSequence([args.seed, index, 1]).generate_state(1)[0])
        np.random.seed(field_seed)  # Existing GRF sampler uses NumPy's global RNG.
        logk = grf.sample()
        k_field = np.exp(logk)
        head, message, success = hydraulic.solve_head(
            k_field, x, y, dh=args.head_gradient, spacing_mode="center_faces"
        )
        if not success:
            raise RuntimeError(f"head solve failed for realization {index}: {message}")
        vx, vy = hydraulic.compute_velocity(
            k_field, x, y, head, output_location="face",
            face_output="full", align_with_modflow=False,
        )
        rng = np.random.default_rng(particle_seed)
        seeds_x = rng.uniform(source[0], source[1], args.num_particles).astype(np.float32)
        seeds_y = rng.uniform(source[2], source[3], args.num_particles).astype(np.float32)
        summary = particle_tracking.particle_tracking_summary({
            "method": "pypar2", "backend": args.taichi_arch,
            "taichi_arch": args.taichi_arch,
            "tracking_mode": "arrival", "stop_mode": "all",
            "save_trajectories": False, "return_fastest": False,
            "x_faces": x_faces, "y_faces": y_faces, "vx": vx, "vy": vy,
            "seeds_x": seeds_x, "seeds_y": seeds_y,
            "right_x": float(x_faces[-1]), "dt": args.dt, "steps": steps,
            "porosity": args.porosity,
            "molecular_diffusion": args.molecular_diffusion,
            "alpha_l": args.alpha_l, "alpha_t": args.alpha_t,
            "snapshot_mode": "physical", "snapshot_times": [args.t_star],
            "seed": particle_seed,
        })
        if "snapshots" not in summary:
            raise RuntimeError(f"PyPar2 returned no snapshot for realization {index}")
        c, active = concentration(summary["snapshots"], x_faces, y_faces, source, args.num_particles)
        filename = f"concentration_{index:04d}.npz"
        np.savez_compressed(args.output_dir / filename, concentration=c,
                            realization=index, field_seed=field_seed,
                            particle_seed=particle_seed, n_active=active)
        files.append(filename)
        print(f"[{index + 1}/{args.num_realizations}] {filename}: {active}/{args.num_particles} active", flush=True)
    metadata = vars(args).copy()
    metadata["output_dir"] = str(args.output_dir)
    metadata.update({"lx": 20.0, "integral_scale_y": 1.0,
                     "source_bounds": list(source), "steps": steps,
                     "concentration_units": "relative to expected initial source concentration",
                     "files": files})
    np.savez_compressed(args.output_dir / "grid.npz", x_faces=x_faces, y_faces=y_faces)
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    run(parser().parse_args())
