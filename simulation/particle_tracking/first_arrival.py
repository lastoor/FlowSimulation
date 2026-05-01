"""Grid first-arrival-time computation from backward particle tracking."""

import warnings

import numpy as np

from .pollock_core import (
    STATUS_SOURCE_INITIAL,
    STATUS_SUCCESS,
)
from .pollock_cpu import pollock_source_arrival_times_cpu


def _faces_from_centers(centers, name):
    centers = np.asarray(centers, dtype=np.float32).ravel()
    if centers.size < 2:
        raise ValueError(f"{name} must contain at least 2 cell centers.")
    faces = np.empty(centers.size + 1, dtype=np.float32)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    faces[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    faces[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return faces


def _centers_from_faces(faces, name):
    faces = np.asarray(faces, dtype=np.float32).ravel()
    if faces.size < 2:
        raise ValueError(f"{name} must contain at least 2 face coordinates.")
    return (0.5 * (faces[:-1] + faces[1:])).astype(np.float32)


def _cell_velocity_to_full_faces(vx_cell, vy_cell):
    vx_cell = np.asarray(vx_cell, dtype=np.float32)
    vy_cell = np.asarray(vy_cell, dtype=np.float32)
    if vx_cell.shape != vy_cell.shape:
        raise ValueError("cell-centered vx and vy must have the same shape.")
    if vx_cell.ndim != 2:
        raise ValueError("cell-centered vx and vy must be 2D arrays.")

    ny, nx = vx_cell.shape
    vx_faces = np.empty((ny, nx + 1), dtype=np.float32)
    vy_faces = np.empty((ny + 1, nx), dtype=np.float32)

    vx_faces[:, 1:nx] = 0.5 * (vx_cell[:, :-1] + vx_cell[:, 1:])
    vx_faces[:, 0] = vx_cell[:, 0]
    vx_faces[:, -1] = vx_cell[:, -1]

    vy_faces[1:ny, :] = 0.5 * (vy_cell[:-1, :] + vy_cell[1:, :])
    vy_faces[0, :] = vy_cell[0, :]
    vy_faces[-1, :] = vy_cell[-1, :]
    return vx_faces, vy_faces


def _prepare_pollock_inputs(params):
    logk_field = np.asarray(params["logk_field"])
    source_mask = np.asarray(params["source_mask"], dtype=bool)
    if logk_field.ndim != 2:
        raise ValueError("logk_field must be a 2D array.")
    if source_mask.shape != logk_field.shape:
        raise ValueError("source_mask must have the same shape as logk_field.")
    if not np.any(source_mask):
        raise ValueError("source_mask must contain at least one source cell.")

    ny, nx = logk_field.shape
    velocity_location = params.get("velocity_location", "face")
    if velocity_location not in ("face", "cell"):
        raise ValueError("velocity_location must be either 'face' or 'cell'.")

    if velocity_location == "face":
        x_faces = np.asarray(params["x_faces"], dtype=np.float32).ravel()
        y_faces = np.asarray(params["y_faces"], dtype=np.float32).ravel()
        vx_faces = np.asarray(params["vx"], dtype=np.float32)
        vy_faces = np.asarray(params["vy"], dtype=np.float32)
        if x_faces.size != nx + 1 or y_faces.size != ny + 1:
            raise ValueError("x_faces/y_faces must match logk_field dimensions.")
        if vx_faces.shape != (ny, nx + 1):
            raise ValueError("face-centered vx shape must be (ny, nx + 1).")
        if vy_faces.shape != (ny + 1, nx):
            raise ValueError("face-centered vy shape must be (ny + 1, nx).")
        x_centers = _centers_from_faces(x_faces, "x_faces")
        y_centers = _centers_from_faces(y_faces, "y_faces")
    else:
        x_centers = np.asarray(params["x"], dtype=np.float32).ravel()
        y_centers = np.asarray(params["y"], dtype=np.float32).ravel()
        if x_centers.size != nx or y_centers.size != ny:
            raise ValueError("x/y cell-center coordinates must match logk_field dimensions.")
        x_faces = _faces_from_centers(x_centers, "x")
        y_faces = _faces_from_centers(y_centers, "y")
        vx_faces, vy_faces = _cell_velocity_to_full_faces(params["vx"], params["vy"])
        if vx_faces.shape != (ny, nx + 1) or vy_faces.shape != (ny + 1, nx):
            raise ValueError("cell-centered vx/vy must match logk_field dimensions.")

    X, Y = np.meshgrid(x_centers, y_centers)
    seeds_x = X.astype(np.float32).ravel()
    seeds_y = Y.astype(np.float32).ravel()

    return {
        "shape": (ny, nx),
        "x_faces": x_faces,
        "y_faces": y_faces,
        "vx": (-vx_faces).astype(np.float32),
        "vy": (-vy_faces).astype(np.float32),
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "source_mask": source_mask.astype(np.int32),
        "velocity_location": velocity_location,
    }


def first_arrival_time_grid(params):
    """Compute source first-arrival time at every logK grid cell center.

    Particles are released at cell centers and tracked backward with Pollock
    stepping through the reversed velocity field. Unreached cells are reported
    as ``np.inf``.
    """
    method = params.get("method", "pollock")
    if method != "pollock":
        raise ValueError("first_arrival_time_grid currently supports method='pollock' only.")

    backend = params.get("backend", "gpu")
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'.")

    prepared = _prepare_pollock_inputs(params)
    ny, nx = prepared["shape"]
    max_steps = int(params.get("max_steps", max(1, 4 * nx * ny)))

    taichi_arch = None
    if backend == "gpu":
        try:
            from .pollock_gpu import pollock_source_arrival_times_gpu

            travel_times, status, taichi_arch = pollock_source_arrival_times_gpu(
                prepared["x_faces"],
                prepared["y_faces"],
                prepared["vx"],
                prepared["vy"],
                prepared["seeds_x"],
                prepared["seeds_y"],
                prepared["source_mask"],
                max_steps,
            )
        except Exception as exc:
            if not bool(params.get("fallback_to_cpu", True)):
                raise
            warnings.warn(
                f"GPU Pollock first-arrival tracking failed; falling back to CPU ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = "cpu"
            travel_times, status = pollock_source_arrival_times_cpu(
                prepared["x_faces"],
                prepared["y_faces"],
                prepared["vx"],
                prepared["vy"],
                prepared["seeds_x"],
                prepared["seeds_y"],
                prepared["source_mask"],
                max_steps,
            )
    else:
        travel_times, status = pollock_source_arrival_times_cpu(
            prepared["x_faces"],
            prepared["y_faces"],
            prepared["vx"],
            prepared["vy"],
            prepared["seeds_x"],
            prepared["seeds_y"],
            prepared["source_mask"],
            max_steps,
        )

    arrival_time = np.asarray(travel_times, dtype=np.float32).reshape((ny, nx))
    status_grid = np.asarray(status, dtype=np.int32).reshape((ny, nx))
    reached = (status_grid == STATUS_SUCCESS) | (status_grid == STATUS_SOURCE_INITIAL)
    arrival_time = np.where(reached, arrival_time, np.inf).astype(np.float32)

    result = {
        "arrival_time": arrival_time,
        "status": status_grid,
        "backend": backend,
        "method": method,
        "velocity_location": prepared["velocity_location"],
        "max_steps": max_steps,
    }
    if taichi_arch is not None:
        result["taichi_arch"] = taichi_arch
    return result
