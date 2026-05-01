#!/usr/bin/env python
"""End-to-end first-arrival-time field validation and visualization."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation import grf_generation, hydraulic, particle_tracking
from simulation.particle_tracking.pollock_cpu import (
    _nudge_to_next_cell,
    _pollock_step_with_hitinfo,
    _source_neighbor_from_hit,
    find_cell_index,
)
from simulation.particle_tracking.pollock_core import STATUS_SUCCESS


STATUS_LABELS = {
    0: "source hit",
    1: "stagnation",
    2: "out of bounds",
    4: "max steps",
    5: "source initial",
}


def _faces_from_centers(centers):
    centers = np.asarray(centers, dtype=np.float32).ravel()
    faces = np.empty(centers.size + 1, dtype=np.float32)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    faces[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    faces[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return faces


def _cell_velocity_from_full_faces(vx_face, vy_face):
    vx_cell = 0.5 * (vx_face[:, :-1] + vx_face[:, 1:])
    vy_cell = 0.5 * (vy_face[:-1, :] + vy_face[1:, :])
    return vx_cell, vy_cell


def _trace_backward_path_to_source(x0, y0, x_faces, y_faces, vx_face, vy_face, source_mask, max_steps):
    vx_rev = -np.asarray(vx_face, dtype=np.float32)
    vy_rev = -np.asarray(vy_face, dtype=np.float32)
    x_faces = np.asarray(x_faces, dtype=np.float32)
    y_faces = np.asarray(y_faces, dtype=np.float32)
    source_mask = np.asarray(source_mask, dtype=np.int32)

    curr_x = np.float32(x0)
    curr_y = np.float32(y0)
    curr_t = np.float32(0.0)
    path_x = [float(curr_x)]
    path_y = [float(curr_y)]
    path_t = [float(curr_t)]
    status = STATUS_SUCCESS

    i0 = find_cell_index(y_faces, curr_y)
    j0 = find_cell_index(x_faces, curr_x)
    if i0 == -1 or j0 == -1:
        return np.asarray(path_x), np.asarray(path_y), np.asarray(path_t), 2
    if source_mask[i0, j0] != 0:
        return np.asarray(path_x), np.asarray(path_y), np.asarray(path_t), STATUS_SUCCESS

    for _ in range(int(max_steps)):
        prev_i = find_cell_index(y_faces, curr_y)
        prev_j = find_cell_index(x_faces, curr_x)
        if prev_i == -1 or prev_j == -1:
            status = 2
            break

        next_x, next_y, dt, step_status, hit_axis, hit_index, hit_direction = _pollock_step_with_hitinfo(
            curr_x,
            curr_y,
            x_faces,
            y_faces,
            vx_rev,
            vy_rev,
            x_faces[-1],
        )
        if step_status != STATUS_SUCCESS:
            status = int(step_status)
            break

        curr_t = np.float32(curr_t + dt)
        curr_x = np.float32(next_x)
        curr_y = np.float32(next_y)
        path_x.append(float(curr_x))
        path_y.append(float(curr_y))
        path_t.append(float(curr_t))

        source_i, source_j = _source_neighbor_from_hit(prev_i, prev_j, hit_axis, hit_index, hit_direction)
        if 0 <= source_i < source_mask.shape[0] and 0 <= source_j < source_mask.shape[1]:
            if source_mask[source_i, source_j] != 0:
                status = STATUS_SUCCESS
                break
        else:
            status = 2
            break

        if curr_x <= x_faces[0] or curr_x >= x_faces[-1] or curr_y <= y_faces[0] or curr_y >= y_faces[-1]:
            status = 2
            break

        curr_x, curr_y = _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction)
    else:
        status = 4

    return np.asarray(path_x), np.asarray(path_y), np.asarray(path_t), int(status)


def _resample_polyline(x, y, n_samples=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.empty((0, 2), dtype=float)
    if x.size == 1:
        return np.repeat(np.column_stack([x, y]), n_samples, axis=0)

    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    if arclen[-1] <= 0.0:
        return np.repeat(np.array([[x[0], y[0]]], dtype=float), n_samples, axis=0)
    target = np.linspace(0.0, arclen[-1], n_samples)
    return np.column_stack([np.interp(target, arclen, x), np.interp(target, arclen, y)])


def _path_mean_distance(path_a, path_b):
    if path_a.shape[0] == 0 or path_b.shape[0] == 0:
        return np.inf
    n = min(path_a.shape[0], path_b.shape[0])
    a = _resample_polyline(path_a[:, 0], path_a[:, 1], n)
    b = _resample_polyline(path_b[:, 0], path_b[:, 1], n)
    return float(np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1))))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Validate and visualize the first-arrival-time field.")
    parser.add_argument("--field-size", type=int, default=80)
    parser.add_argument("--hurst", type=float, default=0.25)
    parser.add_argument("--std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dh", type=float, default=0.01)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    np.random.seed(args.seed)

    nx = ny = int(args.field_size)
    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    x_faces = _faces_from_centers(x)
    y_faces = _faces_from_centers(y)

    x0 = max(0, nx // 20 - 1)
    y_pad = max(1, ny // 20 - 1)
    y0 = y_pad
    y1 = ny - y_pad - 1
    target_j = nx - max(2, nx // 20) - 1
    target_x = float(x[target_j])

    c = grf_generation.compute_c(nx, args.hurst, args.std)
    fbm = grf_generation.fBmGenerator(c=c, alpha=2.0 * args.hurst, num_pt=nx, minpadding=0)
    logk_field, msg = grf_generation.generate_logk_field(fbm)
    if logk_field is None:
        raise RuntimeError(msg)
    logk_field = np.asarray(logk_field, dtype=np.float64)
    k_field = np.exp(logk_field)

    h, head_msg, is_success = hydraulic.solve_head(
        k_field,
        x,
        y,
        dh=args.dh,
        rtol=1e-10,
        atol=1e-10,
        max_it=10000,
    )
    if not is_success:
        raise RuntimeError(f"Head solve failed: {head_msg}")

    vx_face, vy_face = hydraulic.compute_velocity(
        k_field,
        x,
        y,
        h,
        output_location="face",
        face_output="full",
        align_with_modflow=False,
    )
    vx_face = np.asarray(vx_face, dtype=np.float32)
    vy_face = np.asarray(vy_face, dtype=np.float32)
    vx_cell, vy_cell = _cell_velocity_from_full_faces(vx_face, vy_face)

    source_mask = np.zeros((ny, nx), dtype=bool)
    source_mask[y0 : y1 + 1, x0] = True
    target_rows = np.arange(1, ny - 1, dtype=int)

    max_steps = 4 * nx * ny
    first_arrival = particle_tracking.first_arrival_time_grid(
        {
            "logk_field": logk_field,
            "source_mask": source_mask,
            "velocity_location": "face",
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx_face,
            "vy": vy_face,
            "backend": args.backend,
            "fallback_to_cpu": True,
            "max_steps": max_steps,
        }
    )
    arrival_time = first_arrival["arrival_time"]
    status_grid = np.asarray(first_arrival["status"], dtype=np.int32)

    target_times = arrival_time[target_rows, target_j]
    finite = np.isfinite(target_times)
    if not np.any(finite):
        raise RuntimeError("No finite first-arrival time found in the target region.")
    local_idx = int(np.argmin(np.where(finite, target_times, np.inf)))
    target_i = int(target_rows[local_idx])
    target_min_time = float(target_times[local_idx])

    seeds_y = y[y0 : y1 + 1].astype(np.float32)
    source_exit_x = np.float32(x_faces[x0 + 1])
    seeds_x = np.full(seeds_y.size, source_exit_x, dtype=np.float32)
    pollock_summary = particle_tracking.particle_tracking_summary(
        {
            "backend": "cpu",
            "method": "pollock",
            "tracking_mode": "arrival",
            "stop_mode": "first",
            "save_trajectories": True,
            "return_fastest": True,
            "x_faces": x_faces,
            "y_faces": y_faces,
            "vx": vx_face,
            "vy": vy_face,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": target_x,
            "max_points": max_steps + 2,
            "max_steps": max_steps,
        }
    )
    pollock_fastest = pollock_summary.get("fastest")
    if pollock_fastest is None:
        raise RuntimeError("Pollock reference did not find a fastest path.")

    bt_x, bt_y, bt_t, bt_status = _trace_backward_path_to_source(
        x[target_j],
        y[target_i],
        x_faces,
        y_faces,
        vx_face,
        vy_face,
        source_mask,
        max_steps,
    )
    if bt_status != STATUS_SUCCESS:
        raise RuntimeError(f"Backtracking from target-min cell failed with status={bt_status}.")

    backtrack_forward = np.column_stack([bt_x[::-1], bt_y[::-1]])
    pollock_path = np.column_stack([pollock_fastest["x"], pollock_fastest["y"]])
    time_diff = abs(target_min_time - float(pollock_fastest["travel_time"]))
    path_mean_distance = _path_mean_distance(pollock_path, backtrack_forward)
    tolerance = max(1e-3, 1e-2 * float(pollock_fastest["travel_time"]))

    print(msg)
    print(head_msg)
    print(f"first_arrival backend: {first_arrival['backend']}")
    print("first_arrival status counts:")
    status_values, status_counts = np.unique(status_grid, return_counts=True)
    for code, count in zip(status_values, status_counts):
        label = STATUS_LABELS.get(int(code), "unknown")
        print(f"  status {int(code):2d} ({label}): {int(count)}")
    print(f"target cell: row={target_i}, col={target_j}, x={x[target_j]:.3f}, y={y[target_i]:.3f}")
    print(f"target-min first-arrival time: {target_min_time:.8g}")
    print(f"Pollock fastest travel time: {float(pollock_fastest['travel_time']):.8g}")
    print(f"absolute time difference: {time_diff:.8g} (tolerance {tolerance:.8g})")
    print(f"mean resampled path distance: {path_mean_distance:.8g}")
    if time_diff > tolerance:
        print("WARNING: first-arrival target minimum and Pollock fastest time differ beyond tolerance.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.out_dir / "first_arrival_time_field.png"
    npz_path = args.out_dir / "first_arrival_time_field.npz"

    masked_arrival = np.ma.masked_invalid(np.where(np.isfinite(arrival_time), arrival_time, np.nan))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(masked_arrival, origin="lower", cmap="magma", aspect="equal")
    plt.colorbar(im, ax=ax, label="first arrival time")
    ax.streamplot(
        x,
        y,
        vx_cell,
        vy_cell,
        color="white",
        linewidth=0.55,
        density=1.2,
        arrowsize=0.65,
        minlength=0.1,
    )
    ax.contour(source_mask.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=1.8)
    ax.axvline(x[target_j], color="#7CFC00", linewidth=1.4, linestyle=":", label="target column")
    ax.plot(
        pollock_fastest["x"],
        pollock_fastest["y"],
        color="#00E5FF",
        linewidth=2.2,
        label=f"Pollock fastest, t={float(pollock_fastest['travel_time']):.3g}",
    )
    ax.plot(
        backtrack_forward[:, 0],
        backtrack_forward[:, 1],
        color="#7CFC00",
        linewidth=2.0,
        linestyle="--",
        label=f"FAT target-min backtrack, t={target_min_time:.3g}",
    )
    ax.scatter([x[target_j]], [y[target_i]], color="#7CFC00", edgecolor="black", zorder=5, label="target min")
    ax.set_xlim(x_faces[0], x_faces[-1])
    ax.set_ylim(y_faces[0], y_faces[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "First arrival time field vs Pollock fastest path\n"
        f"|dt|={time_diff:.3g}, path mean distance={path_mean_distance:.3g}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        logk_field=logk_field,
        head=h,
        arrival_time=arrival_time,
        status=first_arrival["status"],
        vx_face=vx_face,
        vy_face=vy_face,
        pollock_x=np.asarray(pollock_fastest["x"]),
        pollock_y=np.asarray(pollock_fastest["y"]),
        pollock_t=np.asarray(pollock_fastest["t"]),
        pollock_travel_time=np.asarray(float(pollock_fastest["travel_time"])),
        backtrack_x=bt_x,
        backtrack_y=bt_y,
        backtrack_t=bt_t,
        target_min_time=np.asarray(target_min_time),
        time_diff=np.asarray(time_diff),
        path_mean_distance=np.asarray(path_mean_distance),
        source_mask=source_mask,
        target_i=np.asarray(target_i),
        target_j=np.asarray(target_j),
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {npz_path}")


if __name__ == "__main__":
    main()
