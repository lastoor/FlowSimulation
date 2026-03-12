import numpy as np
from numba import njit


STATUS_SUCCESS = np.int32(0)
STATUS_STAGNATION = np.int32(1)
STATUS_OUT_OF_BOUNDS = np.int32(2)
STATUS_MAX_STEPS = np.int32(4)

_HUGE_DT = 1e30
_EPS = 1e-12


@njit
def find_cell_index(grid, val):
    """Find the index i such that grid[i] <= val < grid[i+1]."""
    if val < grid[0] or val > grid[-1]:
        return -1
    if val == grid[-1]:
        return len(grid) - 2

    low = 0
    high = len(grid) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if val >= grid[mid]:
            low = mid
        else:
            high = mid
    return low


@njit
def pollock_step(x, y, x_faces, y_faces, vx_faces, vy_faces):
    """
    Compute a single analytical step through one cell using full face velocities.

    Parameters
    ----------
    x, y : float
        Current particle position.
    x_faces, y_faces : 1D arrays
        Face coordinates.
    vx_faces : 2D array, shape (ny, nx+1)
    vy_faces : 2D array, shape (ny+1, nx)

    Returns
    -------
    x_new, y_new, dt, status
    """
    j = find_cell_index(x_faces, x)
    i = find_cell_index(y_faces, y)

    if i == -1 or j == -1:
        return x, y, 0.0, STATUS_OUT_OF_BOUNDS

    x1 = x_faces[j]
    x2 = x_faces[j + 1]
    y1 = y_faces[i]
    y2 = y_faces[i + 1]
    dx = x2 - x1
    dy = y2 - y1
    if dx <= 0.0 or dy <= 0.0:
        return x, y, 0.0, STATUS_STAGNATION

    v1x = vx_faces[i, j]
    v2x = vx_faces[i, j + 1]
    v1y = vy_faces[i, j]
    v2y = vy_faces[i + 1, j]

    ax = (v2x - v1x) / dx
    ay = (v2y - v1y) / dy
    vpx = v1x + ax * (x - x1)
    vpy = v1y + ay * (y - y1)

    dtx = _HUGE_DT
    if vpx > _EPS and v2x > _EPS:
        if abs(ax) > _EPS:
            dtx = np.log(v2x / vpx) / ax
        else:
            dtx = (x2 - x) / vpx
    elif vpx < -_EPS and v1x < -_EPS:
        if abs(ax) > _EPS:
            dtx = np.log(v1x / vpx) / ax
        else:
            dtx = (x1 - x) / vpx

    dty = _HUGE_DT
    if vpy > _EPS and v2y > _EPS:
        if abs(ay) > _EPS:
            dty = np.log(v2y / vpy) / ay
        else:
            dty = (y2 - y) / vpy
    elif vpy < -_EPS and v1y < -_EPS:
        if abs(ay) > _EPS:
            dty = np.log(v1y / vpy) / ay
        else:
            dty = (y1 - y) / vpy

    dt_exit = min(dtx, dty)
    if dt_exit == _HUGE_DT or dt_exit <= _EPS or not np.isfinite(dt_exit):
        return x, y, 0.0, STATUS_STAGNATION

    if abs(ax) > _EPS:
        x_new = x1 + (vpx * np.exp(ax * dt_exit) - v1x) / ax
    else:
        x_new = x + vpx * dt_exit

    if abs(ay) > _EPS:
        y_new = y1 + (vpy * np.exp(ay * dt_exit) - v1y) / ay
    else:
        y_new = y + vpy * dt_exit

    if dtx < dty:
        x_new = x2 if vpx > 0.0 else x1
    else:
        y_new = y2 if vpy > 0.0 else y1

    return x_new, y_new, dt_exit, STATUS_SUCCESS


@njit
def trace_pathline(x0, y0, x_faces, y_faces, vx_faces, vy_faces, max_steps=1000):
    """Trace one particle until it exits, stagnates, or reaches max_steps."""
    path = np.zeros((max_steps + 1, 2), dtype=np.float32)
    times = np.zeros(max_steps + 1, dtype=np.float32)

    path[0, 0] = x0
    path[0, 1] = y0
    curr_x = x0
    curr_y = y0
    total_time = 0.0

    steps = 0
    final_status = STATUS_MAX_STEPS
    for s in range(1, max_steps + 1):
        curr_x, curr_y, dt, status = pollock_step(
            curr_x, curr_y, x_faces, y_faces, vx_faces, vy_faces
        )
        total_time += dt
        path[s, 0] = curr_x
        path[s, 1] = curr_y
        times[s] = total_time
        steps = s

        if status != STATUS_SUCCESS:
            final_status = status
            break

        if curr_x <= x_faces[0] or curr_x >= x_faces[-1] or curr_y <= y_faces[0] or curr_y >= y_faces[-1]:
            final_status = STATUS_SUCCESS
            break
    else:
        final_status = STATUS_MAX_STEPS

    return path[:steps + 1], times[:steps + 1], final_status


@njit
def _trace_one_particle(x0, y0, x_faces, y_faces, vx_faces, vy_faces, right_x, max_points, max_steps):
    px = np.zeros(max_points, dtype=np.float32)
    py = np.zeros(max_points, dtype=np.float32)
    pt = np.zeros(max_points, dtype=np.float32)

    px[0] = x0
    py[0] = y0
    curr_x = x0
    curr_y = y0
    curr_t = 0.0
    length = 1
    status = STATUS_MAX_STEPS

    if curr_x >= right_x:
        status = STATUS_SUCCESS
        return px, py, pt, length, status

    for _ in range(max_steps):
        if length >= max_points:
            status = STATUS_MAX_STEPS
            break

        next_x, next_y, dt, step_status = pollock_step(
            curr_x, curr_y, x_faces, y_faces, vx_faces, vy_faces
        )
        if step_status != STATUS_SUCCESS:
            status = step_status
            break

        curr_t += dt
        curr_x = next_x
        curr_y = next_y
        px[length] = curr_x
        py[length] = curr_y
        pt[length] = curr_t
        length += 1

        if curr_x >= right_x:
            status = STATUS_SUCCESS
            break
        if curr_x <= x_faces[0] or curr_x >= x_faces[-1] or curr_y <= y_faces[0] or curr_y >= y_faces[-1]:
            status = STATUS_SUCCESS
            break
    else:
        status = STATUS_MAX_STEPS

    return px, py, pt, length, status


def particle_tracking_pollock(params):
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

    n_particles = seeds_x.size
    paths_x = np.zeros((n_particles, max_points), dtype=np.float32)
    paths_y = np.zeros((n_particles, max_points), dtype=np.float32)
    paths_t = np.zeros((n_particles, max_points), dtype=np.float32)
    lengths = np.zeros(n_particles, dtype=np.int32)
    status = np.zeros(n_particles, dtype=np.int32)

    for i in range(n_particles):
        px, py, pt, length, st = _trace_one_particle(
            seeds_x[i],
            seeds_y[i],
            x_faces,
            y_faces,
            vx,
            vy,
            right_x,
            max_points,
            max_steps,
        )
        paths_x[i, :] = px
        paths_y[i, :] = py
        paths_t[i, :] = pt
        lengths[i] = length
        status[i] = st

    return paths_x, paths_y, paths_t, lengths, status
