"""Taichi-based GPU Pollock particle-tracking summary backend."""

import warnings

import numpy as np

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
from .pollock_cpu import (
    _trace_one_particle,
)
from .utils import build_fastest_result_from_trajectory, fastest_particle_id, summarize_arrivals

_ti_initialized = False
_tracker_cls = None


def _ensure_ti_init():
    global _ti_initialized
    if _ti_initialized:
        return
    import taichi as ti

    try:
        ti.init(arch=ti.gpu, verbose=False)
    except Exception as exc:
        warnings.warn(
            f"Taichi GPU initialization failed; falling back to Taichi CPU backend ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        ti.init(arch=ti.cpu, verbose=False)
    _ti_initialized = True


def _make_pollock_tracker():
    global _tracker_cls
    if _tracker_cls is not None:
        return _tracker_cls
    import taichi as ti

    @ti.data_oriented
    class PollockTracking2D:
        def __init__(self, x_faces, y_faces, vx_faces, vy_faces, seeds_x, seeds_y, right_x, max_steps, snapshot_times=None):
            x_faces = np.asarray(x_faces, dtype=np.float32).ravel()
            y_faces = np.asarray(y_faces, dtype=np.float32).ravel()
            vx_faces = np.asarray(vx_faces, dtype=np.float32)
            vy_faces = np.asarray(vy_faces, dtype=np.float32)
            seeds_x = np.asarray(seeds_x, dtype=np.float32).ravel()
            seeds_y = np.asarray(seeds_y, dtype=np.float32).ravel()
            if seeds_x.size != seeds_y.size:
                raise ValueError("seeds_x and seeds_y must have the same length.")

            self.nx = x_faces.size - 1
            self.ny = y_faces.size - 1
            self.n_particles = seeds_x.size
            self.max_steps = int(max_steps)
            self.right_x = np.float32(right_x)

            self.x_faces = ti.field(dtype=ti.f32, shape=x_faces.size)
            self.y_faces = ti.field(dtype=ti.f32, shape=y_faces.size)
            self.vx_faces = ti.field(dtype=ti.f32, shape=vx_faces.shape)
            self.vy_faces = ti.field(dtype=ti.f32, shape=vy_faces.shape)

            self.x_faces.from_numpy(x_faces)
            self.y_faces.from_numpy(y_faces)
            self.vx_faces.from_numpy(vx_faces)
            self.vy_faces.from_numpy(vy_faces)

            self.seed_x = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.seed_y = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.pos_x = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.pos_y = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.elapsed = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.final_times = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.status = ti.field(dtype=ti.i32, shape=self.n_particles)
            self.active = ti.field(dtype=ti.i32, shape=self.n_particles)
            self.seed_x.from_numpy(seeds_x)
            self.seed_y.from_numpy(seeds_y)
            self.pos_x.from_numpy(seeds_x)
            self.pos_y.from_numpy(seeds_y)
            self._seeds_x = seeds_x.copy()
            self._seeds_y = seeds_y.copy()
            self._x_faces_np = x_faces
            self._y_faces_np = y_faces
            self._vx_faces_np = vx_faces
            self._vy_faces_np = vy_faces

            # Snapshot fields
            if snapshot_times is not None and len(snapshot_times) > 0:
                snap_np = np.sort(np.asarray(snapshot_times, dtype=np.float32).ravel())
                self.n_snaps = snap_np.size
                self.snap_times = ti.field(dtype=ti.f32, shape=self.n_snaps)
                self.snap_x = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_snaps))
                self.snap_y = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_snaps))
                self.snap_times.from_numpy(snap_np)
                self.snap_x.from_numpy(np.full((self.n_particles, self.n_snaps), np.nan, dtype=np.float32))
                self.snap_y.from_numpy(np.full((self.n_particles, self.n_snaps), np.nan, dtype=np.float32))
            else:
                self.n_snaps = 0
                self.snap_times = ti.field(dtype=ti.f32, shape=1)
                self.snap_x = ti.field(dtype=ti.f32, shape=(1, 1))
                self.snap_y = ti.field(dtype=ti.f32, shape=(1, 1))

        @ti.func
        def _find_cell_index(self, grid, n, val):
            idx = -1
            if val >= grid[0] and val <= grid[n - 1]:
                if val == grid[n - 1]:
                    idx = n - 2
                else:
                    low = 0
                    high = n - 1
                    while high - low > 1:
                        mid = (low + high) // 2
                        if val >= grid[mid]:
                            low = mid
                        else:
                            high = mid
                    idx = low
            return idx

        @ti.func
        def _local_nudge(self, span, coord):
            scale = ti.abs(span) * POLLOCK_NUDGE_REL
            floor = ti.max(ti.abs(coord), 1.0) * POLLOCK_NUDGE_F32
            return ti.max(scale, floor)

        @ti.func
        def _nudge_to_next_cell(self, curr_x, curr_y, hit_axis, hit_index, hit_direction):
            if hit_axis == 0:
                if 0 < hit_index < self.x_faces.shape[0] - 1:
                    span = ti.select(
                        hit_direction > 0,
                        self.x_faces[hit_index] - self.x_faces[hit_index - 1],
                        self.x_faces[hit_index + 1] - self.x_faces[hit_index],
                    )
                    curr_x += ti.cast(hit_direction, ti.f32) * self._local_nudge(span, curr_x)
            elif hit_axis == 1:
                if 0 < hit_index < self.y_faces.shape[0] - 1:
                    span = ti.select(
                        hit_direction > 0,
                        self.y_faces[hit_index] - self.y_faces[hit_index - 1],
                        self.y_faces[hit_index + 1] - self.y_faces[hit_index],
                    )
                    curr_y += ti.cast(hit_direction, ti.f32) * self._local_nudge(span, curr_y)
            return curr_x, curr_y

        @ti.func
        def _pollock_step(self, x, y):
            x_new = x
            y_new = y
            dt_exit = 0.0
            status = STATUS_SUCCESS
            hit_axis = -1
            hit_index = -1
            hit_direction = 0

            j = self._find_cell_index(self.x_faces, self.x_faces.shape[0], x)
            i = self._find_cell_index(self.y_faces, self.y_faces.shape[0], y)
            if i == -1 or j == -1:
                status = STATUS_OUT_OF_BOUNDS

            x1 = 0.0
            x2 = 0.0
            y1 = 0.0
            y2 = 0.0
            dx = 0.0
            dy = 0.0
            if status == STATUS_SUCCESS:
                x1 = self.x_faces[j]
                x2 = self.x_faces[j + 1]
                y1 = self.y_faces[i]
                y2 = self.y_faces[i + 1]
                dx = x2 - x1
                dy = y2 - y1
                if dx <= 0.0 or dy <= 0.0:
                    status = STATUS_STAGNATION

            if status == STATUS_SUCCESS:
                v1x = self.vx_faces[i, j]
                v2x = self.vx_faces[i, j + 1]
                v1y = self.vy_faces[i, j]
                v2y = self.vy_faces[i + 1, j]

                ax = (v2x - v1x) / dx
                ay = (v2y - v1y) / dy
                vpx = v1x + ax * (x - x1)
                vpy = v1y + ay * (y - y1)

                target_x = x2
                dtx = POLLOCK_HUGE_DT
                if vpx > POLLOCK_EPS and v2x > POLLOCK_EPS:
                    target_x = ti.min(x2, self.right_x)
                    v_target = v1x + ax * (target_x - x1)
                    if ti.abs(ax) > POLLOCK_EPS:
                        dtx = ti.log(v_target / vpx) / ax
                    else:
                        dtx = (target_x - x) / vpx
                elif vpx < -POLLOCK_EPS and v1x < -POLLOCK_EPS:
                    if ti.abs(ax) > POLLOCK_EPS:
                        dtx = ti.log(v1x / vpx) / ax
                    else:
                        dtx = (x1 - x) / vpx

                dty = POLLOCK_HUGE_DT
                if vpy > POLLOCK_EPS and v2y > POLLOCK_EPS:
                    if ti.abs(ay) > POLLOCK_EPS:
                        dty = ti.log(v2y / vpy) / ay
                    else:
                        dty = (y2 - y) / vpy
                elif vpy < -POLLOCK_EPS and v1y < -POLLOCK_EPS:
                    if ti.abs(ay) > POLLOCK_EPS:
                        dty = ti.log(v1y / vpy) / ay
                    else:
                        dty = (y1 - y) / vpy

                dt_exit = ti.min(dtx, dty)
                if dt_exit == POLLOCK_HUGE_DT or dt_exit <= POLLOCK_EPS or ti.math.isnan(dt_exit):
                    status = STATUS_STAGNATION

                if status == STATUS_SUCCESS:
                    x_new = x + vpx * dt_exit
                    y_new = y + vpy * dt_exit
                    if ti.abs(ax) > POLLOCK_EPS:
                        x_new = x1 + (vpx * ti.exp(ax * dt_exit) - v1x) / ax
                    if ti.abs(ay) > POLLOCK_EPS:
                        y_new = y1 + (vpy * ti.exp(ay * dt_exit) - v1y) / ay

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

            return x_new, y_new, dt_exit, status, hit_axis, hit_index, hit_direction

        @ti.kernel
        def init_state(self):
            for i in range(self.n_particles):
                self.pos_x[i] = self.seed_x[i]
                self.pos_y[i] = self.seed_y[i]
                self.elapsed[i] = 0.0
                self.final_times[i] = POLLOCK_HUGE_DT
                self.status[i] = STATUS_MAX_STEPS
                self.active[i] = 1
                if self.pos_x[i] >= self.right_x:
                    self.final_times[i] = 0.0
                    self.status[i] = STATUS_SUCCESS
                    self.active[i] = 0

        @ti.func
        def _capture_snapshots(self, i: ti.i32, snap_idx: ti.i32, prev_x: ti.f32, prev_y: ti.f32, prev_t: ti.f32, curr_x: ti.f32, curr_y: ti.f32, curr_t: ti.f32) -> ti.i32:
            """Capture any snapshot times that fall within [prev_t, curr_t]."""
            idx = snap_idx
            while idx < self.n_snaps and self.snap_times[idx] <= curr_t:
                t_s = self.snap_times[idx]
                if t_s >= prev_t:
                    dt_step = curr_t - prev_t + 1e-30
                    alpha = (t_s - prev_t) / dt_step
                    self.snap_x[i, idx] = prev_x + alpha * (curr_x - prev_x)
                    self.snap_y[i, idx] = prev_y + alpha * (curr_y - prev_y)
                idx += 1
            return idx

        @ti.kernel
        def track_all(self):
            for i in range(self.n_particles):
                steps = 0
                snap_idx = 0
                while self.active[i] == 1 and steps < self.max_steps:
                    prev_x = self.pos_x[i]
                    prev_y = self.pos_y[i]
                    prev_t = self.elapsed[i]
                    next_x, next_y, dt, step_status, hit_axis, hit_index, hit_direction = self._pollock_step(prev_x, prev_y)

                    if step_status != STATUS_SUCCESS:
                        self.status[i] = step_status
                        self.final_times[i] = prev_t
                        self.active[i] = 0
                    else:
                        curr_t = prev_t + dt
                        curr_x = next_x
                        curr_y = next_y

                        # Capture snapshots in [prev_t, curr_t]
                        if self.n_snaps > 0 and snap_idx < self.n_snaps:
                            snap_idx = self._capture_snapshots(i, snap_idx, prev_x, prev_y, prev_t, curr_x, curr_y, curr_t)

                        self.pos_x[i] = curr_x
                        self.pos_y[i] = curr_y
                        self.elapsed[i] = curr_t

                        if curr_x >= self.right_x:
                            self.status[i] = STATUS_SUCCESS
                            self.final_times[i] = curr_t
                            self.active[i] = 0
                        elif (
                            curr_x <= self.x_faces[0]
                            or curr_x >= self.x_faces[self.x_faces.shape[0] - 1]
                            or curr_y <= self.y_faces[0]
                            or curr_y >= self.y_faces[self.y_faces.shape[0] - 1]
                        ):
                            self.status[i] = STATUS_SUCCESS
                            self.final_times[i] = curr_t
                            self.active[i] = 0
                        else:
                            nudged_x, nudged_y = self._nudge_to_next_cell(curr_x, curr_y, hit_axis, hit_index, hit_direction)
                            self.pos_x[i] = nudged_x
                            self.pos_y[i] = nudged_y
                    steps += 1

                if self.active[i] == 1:
                    self.status[i] = STATUS_MAX_STEPS
                    self.final_times[i] = self.elapsed[i]
                    self.active[i] = 0

        def run(self):
            self.init_state()
            self.track_all()
            import taichi as ti

            ti.sync()
            travel_times = self.final_times.to_numpy()
            status = self.status.to_numpy()
            if self.n_snaps > 0:
                snap_x = self.snap_x.to_numpy()
                snap_y = self.snap_y.to_numpy()
            else:
                snap_x = None
                snap_y = None
            return travel_times, status, snap_x, snap_y

        def replay_selected(self, particle_ids, right_x, max_points, max_steps):
            trajectories = []
            for pid in np.asarray(particle_ids, dtype=np.int32).ravel():
                px, py, pt, length, _ = _trace_one_particle(
                    self._seeds_x[int(pid)],
                    self._seeds_y[int(pid)],
                    self._x_faces_np,
                    self._y_faces_np,
                    self._vx_faces_np,
                    self._vy_faces_np,
                    np.float32(right_x),
                    int(max_points),
                    int(max_steps),
                )
                trajectories.append(
                    {
                        "particle_id": int(pid),
                        "x": px[:length].copy(),
                        "y": py[:length].copy(),
                        "t": pt[:length].copy(),
                    }
                )
            return trajectories

    _tracker_cls = PollockTracking2D
    return _tracker_cls


def _resolve_snapshot_times(snapshot_mode, raw_snapshot_times, travel_times):
    """Resolve snapshot times from user parameters into absolute physical times.

    Parameters
    ----------
    snapshot_mode : str
        One of ``"physical"``, ``"dt_interval"``, ``"fraction_of_first_arrival"``.
    raw_snapshot_times : array-like or scalar
        For ``"physical"``: 1-D array of absolute times.
        For ``"dt_interval"``: scalar dt.
        For ``"fraction_of_first_arrival"``: 1-D array of fractions (values > 1.0 allowed).
    travel_times : ndarray
        Per-particle travel times (NaN for unsuccessful particles).

    Returns
    -------
    ndarray (T,) of sorted absolute snapshot times, or None if resolution fails.
    """
    if snapshot_mode == "physical":
        return np.sort(np.asarray(raw_snapshot_times, dtype=np.float32).ravel())

    valid_times = travel_times[np.isfinite(travel_times)]
    if valid_times.size == 0:
        return None

    if snapshot_mode == "dt_interval":
        dt = float(raw_snapshot_times)
        t_max = float(np.max(valid_times))
        if dt <= 0 or t_max <= 0:
            return None
        return np.arange(dt, t_max + dt * 0.5, dt, dtype=np.float32)

    if snapshot_mode == "fraction_of_first_arrival":
        t_first = float(np.min(valid_times))
        if t_first <= 0:
            return None
        fractions = np.sort(np.asarray(raw_snapshot_times, dtype=np.float32).ravel())
        return (fractions * t_first).astype(np.float32)

    raise ValueError(f"Unknown snapshot_mode: {snapshot_mode!r}. "
                     "Must be 'physical', 'dt_interval', or 'fraction_of_first_arrival'.")


def particle_tracking_pollock_gpu_summary(params):
    """Taichi-based GPU Pollock summary backend."""
    tracking_mode = params.get("tracking_mode", "arrival")
    if tracking_mode not in ("trajectory", "arrival"):
        raise ValueError("tracking_mode must be 'trajectory' or 'arrival'.")

    stop_mode = params.get("stop_mode", "first")
    arrival_fraction = float(params.get("arrival_fraction", 1.0))
    save_trajectories = bool(params.get("save_trajectories", tracking_mode == "trajectory"))
    return_fastest = bool(params.get("return_fastest", stop_mode == "first"))
    if tracking_mode == "trajectory" and stop_mode != "first":
        raise NotImplementedError("GPU Pollock trajectory replay currently supports stop_mode='first' only.")
    if save_trajectories and stop_mode not in ("first", "fraction"):
        raise NotImplementedError("GPU Pollock selected-trajectory replay currently supports stop_mode in {'first', 'fraction'} only.")

    x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_points, max_steps = parse_pollock_inputs(params)

    snapshot_mode = params.get("snapshot_mode", None)
    raw_snapshot_times = params.get("snapshot_times", None)

    _ensure_ti_init()
    PollockTracking2D = _make_pollock_tracker()

    # Determine if we need a two-pass approach for snapshot resolution
    resolved_snap_times = None
    if snapshot_mode is not None and raw_snapshot_times is not None:
        if snapshot_mode == "physical":
            # Single pass: snapshot times are known upfront
            resolved_snap_times = _resolve_snapshot_times(snapshot_mode, raw_snapshot_times, None)
        else:
            # Two-pass: first run without snapshots to get arrival times
            tracker_pass1 = PollockTracking2D(x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_steps)
            raw_travel_times_p1, status_p1, _, _ = tracker_pass1.run()
            travel_times_p1 = np.where(status_p1 == STATUS_SUCCESS, raw_travel_times_p1, np.nan).astype(np.float32)
            resolved_snap_times = _resolve_snapshot_times(snapshot_mode, raw_snapshot_times, travel_times_p1)

    # Main tracking run (with snapshot capture if resolved)
    tracker = PollockTracking2D(x_faces, y_faces, vx, vy, seeds_x, seeds_y, right_x, max_steps,
                                snapshot_times=resolved_snap_times)
    raw_travel_times, status, snap_x, snap_y = tracker.run()
    travel_times = np.where(status == STATUS_SUCCESS, raw_travel_times, np.nan).astype(np.float32)
    summary = summarize_arrivals(travel_times, status, stop_mode=stop_mode, arrival_fraction=arrival_fraction)
    summary.update(
        {
            "backend": "gpu",
            "method": "pollock",
            "tracking_mode": tracking_mode,
            "stop_mode": stop_mode,
            "save_trajectories": save_trajectories,
            "return_fastest": return_fastest,
            "status": status,
            "lengths": None,
        }
    )

    if save_trajectories:
        selected_trajectories = tracker.replay_selected(summary["selected_particle_ids"], right_x, max_points, max_steps)
        summary["selected_trajectories"] = selected_trajectories
    else:
        selected_trajectories = []

    if return_fastest and summary["n_arrived"] > 0:
        fastest_id = fastest_particle_id(travel_times)
        if save_trajectories and selected_trajectories:
            fastest_traj = next((traj for traj in selected_trajectories if traj["particle_id"] == fastest_id), None)
            if fastest_traj is None:
                fastest_traj = tracker.replay_selected(np.array([fastest_id], dtype=np.int32), right_x, max_points, max_steps)[0]
            summary["fastest"] = build_fastest_result_from_trajectory(fastest_id, travel_times[fastest_id], fastest_traj)
        else:
            traj = tracker.replay_selected(np.array([fastest_id], dtype=np.int32), right_x, max_points, max_steps)[0]
            summary["fastest"] = build_fastest_result_from_trajectory(fastest_id, travel_times[fastest_id], traj)

    # Attach snapshots
    if resolved_snap_times is not None and snap_x is not None:
        summary["snapshots"] = {
            "snapshot_mode": snapshot_mode,
            "snapshot_times": resolved_snap_times,
            "x": snap_x,
            "y": snap_y,
        }

    return summary
