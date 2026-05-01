"""Shared Pollock constants and input parsing helpers."""

import numpy as np


STATUS_SUCCESS = np.int32(0)
STATUS_STAGNATION = np.int32(1)
STATUS_OUT_OF_BOUNDS = np.int32(2)
STATUS_MAX_STEPS = np.int32(4)
STATUS_SOURCE_INITIAL = np.int32(5)

POLLOCK_HUGE_DT = 1e30
POLLOCK_EPS = 1e-12
POLLOCK_NUDGE_REL = 1e-6
POLLOCK_NUDGE_F32 = 32.0 * np.finfo(np.float32).eps


def parse_pollock_inputs(params):
    """Validate and normalize common Pollock inputs."""
    # Future refinement: support virtual subcell Pollock here by refining
    # x/y faces and resampling vx/vy onto a denser staggered grid before
    # dispatching to the existing CPU/GPU Pollock trackers.
    x_faces = np.asarray(params["x_faces"], dtype=np.float32)
    y_faces = np.asarray(params["y_faces"], dtype=np.float32)
    vx = np.asarray(params["vx"], dtype=np.float32)
    vy = np.asarray(params["vy"], dtype=np.float32)
    seeds_x = np.asarray(params["seeds_x"], dtype=np.float32).ravel()
    seeds_y = np.asarray(params["seeds_y"], dtype=np.float32).ravel()

    if x_faces.ndim != 1 or y_faces.ndim != 1:
        raise ValueError("x_faces and y_faces must be 1D arrays.")
    x_faces = x_faces.ravel()
    y_faces = y_faces.ravel()
    if x_faces.size < 2 or y_faces.size < 2:
        raise ValueError("x_faces and y_faces must contain at least 2 values.")
    if vx.shape != (y_faces.size - 1, x_faces.size):
        raise ValueError("vx shape must be (len(y_faces)-1, len(x_faces)).")
    if vy.shape != (y_faces.size, x_faces.size - 1):
        raise ValueError("vy shape must be (len(y_faces), len(x_faces)-1).")
    if seeds_x.shape != seeds_y.shape:
        raise ValueError("seeds_x and seeds_y must have the same length.")

    right_x = np.float32(params.get("right_x", x_faces[-1]))
    max_points = int(params.get("max_points", 2000))
    max_steps = int(params.get("max_steps", params.get("max_iters", max_points)))
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    return x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_points, max_steps
