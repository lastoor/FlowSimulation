import numpy as np
from numba import njit, prange

from .pollock_core import (
    POLLOCK_EPS,
    POLLOCK_HUGE_DT,
    POLLOCK_NUDGE_F32,
    POLLOCK_NUDGE_REL,
    STATUS_MAX_STEPS,
    STATUS_OUT_OF_BOUNDS,
    STATUS_STAGNATION,
    STATUS_SUCCESS,
    parse_pollock_inputs,
)
from .utils import (
    build_fastest_result_from_arrays,
    build_fastest_result_from_trajectory,
    compute_travel_times,
    extract_particle_trajectories,
    fastest_particle_id,
    summarize_arrivals,
)


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
def _pollock_step_with_hitinfo(x, y, x_faces, y_faces, vx_faces, vy_faces, right_x):
    """Compute a single analytical Pollock step through one cell."""
    j = find_cell_index(x_faces, x)
    i = find_cell_index(y_faces, y)

    if i == -1 or j == -1:
        return x, y, 0.0, STATUS_OUT_OF_BOUNDS, -1, -1, 0

    x1 = x_faces[j]
    x2 = x_faces[j + 1]
    y1 = y_faces[i]
    y2 = y_faces[i + 1]
    dx = x2 - x1
    dy = y2 - y1
    if dx <= 0.0 or dy <= 0.0:
        return x, y, 0.0, STATUS_STAGNATION, -1, -1, 0

    v1x = vx_faces[i, j]
    v2x = vx_faces[i, j + 1]
    v1y = vy_faces[i, j]
    v2y = vy_faces[i + 1, j]

    ax = (v2x - v1x) / dx
    ay = (v2y - v1y) / dy
    vpx = v1x + ax * (x - x1)
    vpy = v1y + ay * (y - y1)

    target_x = x2
    dtx = POLLOCK_HUGE_DT
    if vpx > POLLOCK_EPS and v2x > POLLOCK_EPS:
        target_x = min(x2, right_x)
        v_target = v1x + ax * (target_x - x1)
        if abs(ax) > POLLOCK_EPS:
            dtx = np.log(v_target / vpx) / ax
        else:
            dtx = (target_x - x) / vpx
    elif vpx < -POLLOCK_EPS and v1x < -POLLOCK_EPS:
        if abs(ax) > POLLOCK_EPS:
            dtx = np.log(v1x / vpx) / ax
        else:
            dtx = (x1 - x) / vpx

    dty = POLLOCK_HUGE_DT
    if vpy > POLLOCK_EPS and v2y > POLLOCK_EPS:
        if abs(ay) > POLLOCK_EPS:
            dty = np.log(v2y / vpy) / ay
        else:
            dty = (y2 - y) / vpy
    elif vpy < -POLLOCK_EPS and v1y < -POLLOCK_EPS:
        if abs(ay) > POLLOCK_EPS:
            dty = np.log(v1y / vpy) / ay
        else:
            dty = (y1 - y) / vpy

    dt_exit = min(dtx, dty)
    if dt_exit == POLLOCK_HUGE_DT or dt_exit <= POLLOCK_EPS or not np.isfinite(dt_exit):
        return x, y, 0.0, STATUS_STAGNATION, -1, -1, 0

    if abs(ax) > POLLOCK_EPS:
        x_new = x1 + (vpx * np.exp(ax * dt_exit) - v1x) / ax
    else:
        x_new = x + vpx * dt_exit

    if abs(ay) > POLLOCK_EPS:
        y_new = y1 + (vpy * np.exp(ay * dt_exit) - v1y) / ay
    else:
        y_new = y + vpy * dt_exit

    if dtx < dty:
        if vpx > 0.0:
            x_new = target_x
            hit_index = j + 1
            hit_direction = 1
        else:
            x_new = x1
            hit_index = j
            hit_direction = -1
        hit_axis = 0
    else:
        if vpy > 0.0:
            y_new = y2
            hit_index = i + 1
            hit_direction = 1
        else:
            y_new = y1
            hit_index = i
            hit_direction = -1
        hit_axis = 1

    return x_new, y_new, dt_exit, STATUS_SUCCESS, hit_axis, hit_index, hit_direction


@njit
def pollock_step(x, y, x_faces, y_faces, vx_faces, vy_faces):
    x_new, y_new, dt_exit, status, _, _, _ = _pollock_step_with_hitinfo(
        x, y, x_faces, y_faces, vx_faces, vy_faces, x_faces[-1]
    )
    return x_new, y_new, dt_exit, status


@njit
def _local_nudge(span, coord):
    scale = abs(span) * POLLOCK_NUDGE_REL
    floor = max(abs(coord), 1.0) * POLLOCK_NUDGE_F32
    return max(scale, floor)


@njit
def _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction):
    if hit_axis == 0:
        if 0 < hit_index < x_faces.size - 1:
            span = x_faces[hit_index] - x_faces[hit_index - 1] if hit_direction > 0 else x_faces[hit_index + 1] - x_faces[hit_index]
            curr_x += hit_direction * _local_nudge(span, curr_x)
    elif hit_axis == 1:
        if 0 < hit_index < y_faces.size - 1:
            span = y_faces[hit_index] - y_faces[hit_index - 1] if hit_direction > 0 else y_faces[hit_index + 1] - y_faces[hit_index]
            curr_y += hit_direction * _local_nudge(span, curr_y)
    return curr_x, curr_y


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
        curr_x, curr_y, dt, status, hit_axis, hit_index, hit_direction = _pollock_step_with_hitinfo(
            curr_x, curr_y, x_faces, y_faces, vx_faces, vy_faces, x_faces[-1]
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

        curr_x, curr_y = _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction)
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

        next_x, next_y, dt, step_status, hit_axis, hit_index, hit_direction = _pollock_step_with_hitinfo(
            curr_x, curr_y, x_faces, y_faces, vx_faces, vy_faces, right_x
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

        curr_x, curr_y = _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction)
    else:
        status = STATUS_MAX_STEPS

    return px, py, pt, length, status


@njit(parallel=True)
def _track_arrivals_parallel(seeds_x, seeds_y, x_faces, y_faces, vx, vy, right_x, max_steps):
    n_particles = seeds_x.size
    travel_times = np.zeros(n_particles, dtype=np.float32)
    status = np.zeros(n_particles, dtype=np.int32)

    for i in prange(n_particles):
        curr_x = seeds_x[i]
        curr_y = seeds_y[i]
        curr_t = 0.0

        if curr_x >= right_x:
            travel_times[i] = 0.0
            status[i] = STATUS_SUCCESS
            continue

        for _ in range(max_steps):
            curr_x, curr_y, dt, step_status, hit_axis, hit_index, hit_direction = _pollock_step_with_hitinfo(
                curr_x, curr_y, x_faces, y_faces, vx, vy, right_x
            )

            if step_status != STATUS_SUCCESS:
                status[i] = step_status
                break

            curr_t += dt

            if curr_x >= right_x:
                status[i] = STATUS_SUCCESS
                break
            if curr_x <= x_faces[0] or curr_x >= x_faces[-1] or curr_y <= y_faces[0] or curr_y >= y_faces[-1]:
                status[i] = STATUS_SUCCESS
                break

            curr_x, curr_y = _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction)
        else:
            status[i] = STATUS_MAX_STEPS

        travel_times[i] = curr_t

    return travel_times, status


@njit(parallel=True)
def _track_trajectories_parallel(seeds_x, seeds_y, x_faces, y_faces, vx, vy, right_x, max_points, max_steps):
    n_particles = seeds_x.size
    paths_x = np.zeros((n_particles, max_points), dtype=np.float32)
    paths_y = np.zeros((n_particles, max_points), dtype=np.float32)
    paths_t = np.zeros((n_particles, max_points), dtype=np.float32)
    lengths = np.zeros(n_particles, dtype=np.int32)
    status = np.zeros(n_particles, dtype=np.int32)

    for i in prange(n_particles):
        curr_x = seeds_x[i]
        curr_y = seeds_y[i]
        curr_t = 0.0
        length = 1
        st = STATUS_MAX_STEPS

        paths_x[i, 0] = curr_x
        paths_y[i, 0] = curr_y
        paths_t[i, 0] = curr_t

        if curr_x >= right_x:
            status[i] = STATUS_SUCCESS
            lengths[i] = length
            continue

        for _ in range(max_steps):
            if length >= max_points:
                st = STATUS_MAX_STEPS
                break

            curr_x, curr_y, dt, step_status, hit_axis, hit_index, hit_direction = _pollock_step_with_hitinfo(
                curr_x, curr_y, x_faces, y_faces, vx, vy, right_x
            )

            if step_status != STATUS_SUCCESS:
                st = step_status
                break

            curr_t += dt
            paths_x[i, length] = curr_x
            paths_y[i, length] = curr_y
            paths_t[i, length] = curr_t
            length += 1

            if curr_x >= right_x:
                st = STATUS_SUCCESS
                break
            if curr_x <= x_faces[0] or curr_x >= x_faces[-1] or curr_y <= y_faces[0] or curr_y >= y_faces[-1]:
                st = STATUS_SUCCESS
                break

            curr_x, curr_y = _nudge_to_next_cell(curr_x, curr_y, x_faces, y_faces, hit_axis, hit_index, hit_direction)
        else:
            st = STATUS_MAX_STEPS

        lengths[i] = length
        status[i] = st

    return paths_x, paths_y, paths_t, lengths, status


def particle_tracking_pollock(params):
    x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_points, max_steps = parse_pollock_inputs(params)
    return _track_trajectories_parallel(seeds_x, seeds_y, x_faces, y_faces, vx, vy, right_x, max_points, max_steps)


def particle_tracking_pollock_summary(params):
    """High-level CPU Pollock tracking API with optional arrival-only summaries."""
    tracking_mode = params.get("tracking_mode", "trajectory")
    if tracking_mode not in ("trajectory", "arrival"):
        raise ValueError("tracking_mode must be 'trajectory' or 'arrival'.")

    stop_mode = params.get("stop_mode", "first")
    arrival_fraction = float(params.get("arrival_fraction", 1.0))
    save_trajectories = bool(params.get("save_trajectories", tracking_mode == "trajectory"))
    return_fastest = bool(params.get("return_fastest", stop_mode == "first"))
    x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_points, max_steps = parse_pollock_inputs(params)

    need_dense_trajectories = tracking_mode == "trajectory"
    if need_dense_trajectories:
        px, py, pt, L, st = _track_trajectories_parallel(
            seeds_x, seeds_y, x_faces, y_faces, vx, vy, right_x, max_points, max_steps
        )
        travel_times = compute_travel_times(pt, L, st)
    else:
        px = py = pt = L = None
        travel_times, st = _track_arrivals_parallel(seeds_x, seeds_y, x_faces, y_faces, vx, vy, right_x, max_steps)
        travel_times = np.where(st == STATUS_SUCCESS, travel_times, np.nan).astype(np.float32)

    summary = summarize_arrivals(travel_times, st, stop_mode=stop_mode, arrival_fraction=arrival_fraction)
    summary.update(
        {
            "backend": "cpu",
            "method": "pollock",
            "tracking_mode": tracking_mode,
            "stop_mode": stop_mode,
            "save_trajectories": save_trajectories,
            "return_fastest": return_fastest,
            "status": st,
            "lengths": L,
        }
    )

    if need_dense_trajectories:
        summary["px"] = px
        summary["py"] = py
        summary["pt"] = pt

    if save_trajectories:
        if need_dense_trajectories:
            summary["selected_trajectories"] = extract_particle_trajectories(px, py, pt, L, summary["selected_particle_ids"])
        else:
            selected = []
            for pid in summary["selected_particle_ids"]:
                x_traj, y_traj, t_traj, length, _ = _trace_one_particle(
                    seeds_x[int(pid)],
                    seeds_y[int(pid)],
                    x_faces,
                    y_faces,
                    vx,
                    vy,
                    right_x,
                    max_points,
                    max_steps,
                )
                selected.append(
                    {
                        "particle_id": int(pid),
                        "x": x_traj[:length].copy(),
                        "y": y_traj[:length].copy(),
                        "t": t_traj[:length].copy(),
                    }
                )
            summary["selected_trajectories"] = selected

    if return_fastest and summary["n_arrived"] > 0:
        if need_dense_trajectories:
            summary["fastest"] = build_fastest_result_from_arrays(px, py, pt, L, travel_times)
        else:
            fastest_id = fastest_particle_id(travel_times)
            x_traj, y_traj, t_traj, length, _ = _trace_one_particle(
                seeds_x[fastest_id],
                seeds_y[fastest_id],
                x_faces,
                y_faces,
                vx,
                vy,
                right_x,
                max_points,
                max_steps,
            )
            traj = {
                "particle_id": fastest_id,
                "x": x_traj[:length].copy(),
                "y": y_traj[:length].copy(),
                "t": t_traj[:length].copy(),
            }
            summary["fastest"] = build_fastest_result_from_trajectory(fastest_id, travel_times[fastest_id], traj)

    return summary
