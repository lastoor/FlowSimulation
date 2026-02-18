import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

_ti_initialized = False


def _ensure_ti_init():
    """Initialize Taichi once per process (avoids ti.core which may not exist in all versions)."""
    global _ti_initialized
    if not _ti_initialized:
        ti.init(arch=ti.gpu, verbose=False)
        _ti_initialized = True


def _find_le_index(arr: np.ndarray, val: float) -> int:
    """Largest index i such that arr[i] <= val; clamp to [0, n-2] for interpolation."""
    n = arr.size
    if val <= arr[0]:
        return 0
    if val >= arr[n - 2]:
        return n - 2
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo


def _bilinear_sample_physical(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    nx: int,
    ny: int,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Sample (vx, vy) at physical (x, y) via bilinear interpolation on (xgrid, ygrid)."""
    ix = _find_le_index(xgrid, x)
    iy = _find_le_index(ygrid, y)
    x0, x1 = float(xgrid[ix]), float(xgrid[ix + 1])
    y0, y1 = float(ygrid[iy]), float(ygrid[iy + 1])
    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)
    i00 = iy * nx + ix
    i10 = iy * nx + ix + 1
    i01 = (iy + 1) * nx + ix
    i11 = (iy + 1) * nx + ix + 1
    vx_val = (1 - tx) * (1 - ty) * vx.flat[i00] + tx * (1 - ty) * vx.flat[i10]
    vx_val += (1 - tx) * ty * vx.flat[i01] + tx * ty * vx.flat[i11]
    vy_val = (1 - tx) * (1 - ty) * vy.flat[i00] + tx * (1 - ty) * vy.flat[i10]
    vy_val += (1 - tx) * ty * vy.flat[i01] + tx * ty * vy.flat[i11]
    return vx_val, vy_val


def _resample_velocity_to_uniform(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Resample vx, vy from (x, y) onto a uniform grid over the same extent.
    Returns x_unif (1D nx), y_unif (1D ny), vx_unif (ny, nx), vy_unif (ny, nx),
    and xmin, xmax, ymin, ymax.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    ny, nx = vx.shape
    if x.size != nx or y.size != ny:
        raise ValueError(f"x length {x.size} must equal nx {nx}; y length {y.size} must equal ny {ny}.")
    xmin, xmax = float(x[0]), float(x[-1])
    ymin, ymax = float(y[0]), float(y[-1])
    x_unif = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y_unif = np.linspace(ymin, ymax, ny, dtype=np.float32)
    if np.allclose(x, x_unif, rtol=1e-9, atol=1e-12) and np.allclose(y, y_unif, rtol=1e-9, atol=1e-12):
        return x_unif, y_unif, vx.astype(np.float32), vy.astype(np.float32), xmin, xmax, ymin, ymax
    vx_unif = np.empty((ny, nx), dtype=np.float32)
    vy_unif = np.empty((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            vx_unif[j, i], vy_unif[j, i] = _bilinear_sample_physical(
                x, y, vx, vy, nx, ny, float(x_unif[i]), float(y_unif[j])
            )
    return x_unif, y_unif, vx_unif, vy_unif, xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class SimConfig:
    # Grid and particles
    grid_res: int = 512
    n_particles: int = 1_000_000

    # Time integration
    dt: float = 0.01
    max_steps: int = 5000

    # Domain and release
    source_x: float = 0.05
    target_x: float = 0.95
    y_min: float = 0.1
    y_max: float = 0.9

    # PAR2-like boundary behavior: if a candidate step exits domain in one
    # axis, keep the previous coordinate in that axis.
    closed_boundary: bool = True

    # If closed_boundary is False, particles exiting the domain get this time.
    exit_penalty_time: float = 1e9

    # Optional isotropic random walk term (0 disables diffusion).
    diffusion_sigma: float = 0.0


@ti.data_oriented
class ParticleTracking2D:
    """
    Taichi RK4 particle tracking on a physical (x, y) velocity grid.
    All positions and domain bounds are in physical coordinates.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        vel_grid: np.ndarray,
        seeds_x: np.ndarray,
        seeds_y: np.ndarray,
        right_x: float,
        dt: float = 0.01,
        max_steps: int = 5000,
        exit_penalty_time: float = 1e9,
        y_policy: str = "stop",
        closed_boundary: bool = True,
        diffusion_sigma: float = 0.0,
    ):
        vel_grid = np.asarray(vel_grid, dtype=np.float32)
        if vel_grid.ndim != 3 or vel_grid.shape[2] != 2:
            raise ValueError("vel_grid must have shape (ny, nx, 2).")
        ny, nx = vel_grid.shape[0], vel_grid.shape[1]
        seeds_x = np.asarray(seeds_x, dtype=np.float32).ravel()
        seeds_y = np.asarray(seeds_y, dtype=np.float32).ravel()
        if seeds_x.size != seeds_y.size:
            raise ValueError("seeds_x and seeds_y must have the same length.")
        nP = seeds_x.size

        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.nx = int(nx)
        self.ny = int(ny)
        self.N = nP
        self.right_x = float(right_x)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.exit_penalty_time = float(exit_penalty_time)
        self.closed_boundary = bool(closed_boundary)
        self.diffusion_sigma = float(diffusion_sigma)
        self.y_policy = {"stop": 0, "reflect": 1, "clip": 2}[y_policy]

        self.vel_field = ti.Vector.field(2, dtype=ti.f32, shape=(ny, nx))
        self.vel_field.from_numpy(vel_grid)
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.active = ti.field(dtype=ti.i32, shape=self.N)
        self.elapsed_time = ti.field(dtype=ti.f32, shape=self.N)
        self.final_times = ti.field(dtype=ti.f32, shape=self.N)
        self.active_count = ti.field(dtype=ti.i32, shape=())

        seeds_xy = np.column_stack([seeds_x, seeds_y]).astype(np.float32)
        self.pos.from_numpy(seeds_xy)
        self._seeds_x = seeds_x.copy()
        self._seeds_y = seeds_y.copy()

    @ti.func
    def _gaussian(self):
        u1 = ti.max(ti.random(ti.f32), 1e-7)
        u2 = ti.random(ti.f32)
        r = ti.sqrt(-2.0 * ti.log(u1))
        theta = 2.0 * np.pi * u2
        return r * ti.cos(theta)

    @ti.func
    def _physical_to_index(self, px: ti.f32, py: ti.f32) -> ti.types.vector(4, ti.f32):
        """Map physical (px, py) to continuous indices (ix_float, iy_float, nx, ny)."""
        span_x = self.xmax - self.xmin + 1e-30
        span_y = self.ymax - self.ymin + 1e-30
        u = (px - self.xmin) / span_x * ti.cast(self.nx - 1, ti.f32)
        v = (py - self.ymin) / span_y * ti.cast(self.ny - 1, ti.f32)
        u = ti.min(ti.cast(self.nx - 1, ti.f32), ti.max(0.0, u))
        v = ti.min(ti.cast(self.ny - 1, ti.f32), ti.max(0.0, v))
        return ti.Vector([u, v, ti.cast(self.nx, ti.f32), ti.cast(self.ny, ti.f32)])

    @ti.func
    def bilinear_interp(self, p):
        """Bilinear interpolation at physical position p; returns (vx, vy)."""
        uv = self._physical_to_index(p.x, p.y)
        u, v = uv[0], uv[1]
        x0 = ti.cast(ti.floor(u), ti.i32)
        y0 = ti.cast(ti.floor(v), ti.i32)
        x1 = ti.min(x0 + 1, self.nx - 1)
        y1 = ti.min(y0 + 1, self.ny - 1)
        wx = u - ti.cast(x0, ti.f32)
        wy = v - ti.cast(y0, ti.f32)
        v00 = self.vel_field[y0, x0]
        v10 = self.vel_field[y0, x1]
        v01 = self.vel_field[y1, x0]
        v11 = self.vel_field[y1, x1]
        return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

    @ti.kernel
    def init_particle_state(self):
        """Set active=1, elapsed_time=0, final_times=exit_penalty. (pos already set from host.)"""
        for i in range(self.N):
            self.active[i] = 1
            self.elapsed_time[i] = 0.0
            self.final_times[i] = ti.cast(self.exit_penalty_time, ti.f32)

    @ti.kernel
    def step_rk4(self, dt: ti.f32):
        for i in self.pos:
            if self.active[i] == 1:
                p0 = self.pos[i]
                t0 = self.elapsed_time[i]
                k1 = self.bilinear_interp(p0)
                k2 = self.bilinear_interp(p0 + 0.5 * dt * k1)
                k3 = self.bilinear_interp(p0 + 0.5 * dt * k2)
                k4 = self.bilinear_interp(p0 + dt * k3)
                v_final = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                p1 = p0 + v_final * dt

                if ti.static(True):  # diffusion_sigma check is runtime; branch for clarity
                    sigma_dt = self.diffusion_sigma * ti.sqrt(dt)
                    if sigma_dt > 0.0:
                        p1 += sigma_dt * ti.Vector([self._gaussian(), self._gaussian()])

                if self.closed_boundary:
                    if p1.x < self.xmin or p1.x > self.xmax:
                        p1.x = p0.x
                    if p1.y < self.ymin or p1.y > self.ymax:
                        p1.y = p0.y
                else:
                    if p1.x < self.xmin or p1.x > self.xmax:
                        self.active[i] = 0
                        self.final_times[i] = ti.cast(self.exit_penalty_time, ti.f32)
                        continue
                    if p1.y < self.ymin or p1.y > self.ymax:
                        if self.y_policy == 0:
                            self.active[i] = 0
                            self.final_times[i] = ti.cast(self.exit_penalty_time, ti.f32)
                            continue
                        elif self.y_policy == 1:
                            span = self.ymax - self.ymin
                            Y = self.ymin + (p1.y - self.ymin) % (2.0 * span)
                            if Y < self.ymin:
                                Y += 2.0 * span
                            if Y > self.ymax:
                                Y = self.ymax - (Y - self.ymax)
                            p1.y = Y
                        else:
                            p1.y = ti.max(self.ymin, ti.min(self.ymax, p1.y))

                t1 = t0 + dt
                if p0.x < self.right_x and p1.x >= self.right_x:
                    denom = p1.x - p0.x
                    frac = 1.0
                    if ti.abs(denom) > 1e-8:
                        frac = (self.right_x - p0.x) / denom
                        frac = ti.min(1.0, ti.max(0.0, frac))
                    self.final_times[i] = t0 + frac * dt
                    self.active[i] = 0
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                elif p1.x >= self.right_x:
                    self.final_times[i] = t1
                    self.active[i] = 0
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                else:
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                    ti.atomic_add(self.active_count[None], 1)

    def run_pass1(self):
        self.init_particle_state()
        steps_ran = 0
        for step in range(self.max_steps):
            self.active_count[None] = 0
            self.step_rk4(self.dt)
            steps_ran = step + 1
            if self.active_count[None] == 0:
                break
        ti.sync()
        times_np = self.final_times.to_numpy()
        winner_idx = int(np.argmin(times_np))
        min_time = float(times_np[winner_idx])
        return winner_idx, min_time, times_np

    def _numpy_bilinear_interp_physical(self, px: float, py: float, vel_np: np.ndarray):
        """Bilinear interpolation at physical (px, py). vel_np shape (ny, nx, 2)."""
        span_x = self.xmax - self.xmin + 1e-30
        span_y = self.ymax - self.ymin + 1e-30
        u = (px - self.xmin) / span_x * (self.nx - 1)
        v = (py - self.ymin) / span_y * (self.ny - 1)
        u = max(0.0, min(self.nx - 1, u))
        v = max(0.0, min(self.ny - 1, v))
        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)
        wx = u - x0
        wy = v - y0
        v00 = vel_np[y0, x0]
        v10 = vel_np[y0, x1]
        v01 = vel_np[y1, x0]
        v11 = vel_np[y1, x1]
        vx = (1 - wx) * (1 - wy) * v00[0] + wx * (1 - wy) * v10[0] + (1 - wx) * wy * v01[0] + wx * wy * v11[0]
        vy = (1 - wx) * (1 - wy) * v00[1] + wx * (1 - wy) * v10[1] + (1 - wx) * wy * v01[1] + wx * wy * v11[1]
        return np.array([vx, vy], dtype=np.float32)

    def replay_winner(self, winner_idx: int):
        path_x, path_y = [], []
        p = np.array([float(self._seeds_x[winner_idx]), float(self._seeds_y[winner_idx])], dtype=np.float32)
        vel_np = self.vel_field.to_numpy()
        for _ in range(self.max_steps):
            path_x.append(float(p[0]))
            path_y.append(float(p[1]))
            if p[0] >= self.right_x:
                break
            v = self._numpy_bilinear_interp_physical(float(p[0]), float(p[1]), vel_np)
            p1 = p + v * self.dt
            if self.closed_boundary:
                if p1[0] < self.xmin or p1[0] > self.xmax:
                    p1[0] = p[0]
                if p1[1] < self.ymin or p1[1] > self.ymax:
                    p1[1] = p[1]
            else:
                if p1[1] < self.ymin or p1[1] > self.ymax:
                    if self.y_policy == 0:
                        break
                    elif self.y_policy == 1:
                        span = self.ymax - self.ymin
                        Y = self.ymin + (p1[1] - self.ymin) % (2.0 * span)
                        if Y < self.ymin:
                            Y += 2.0 * span
                        if Y > self.ymax:
                            Y = self.ymax - (Y - self.ymax)
                        p1[1] = Y
                    else:
                        p1[1] = np.clip(p1[1], self.ymin, self.ymax)
            p = p1
        return np.array(path_x, dtype=np.float64), np.array(path_y, dtype=np.float64)


def rk4_fastest_path_params(params: dict) -> dict:
    """
    Run Taichi RK4 fastest-path using the same params dict as particle_tracking_cpu.

    Expects physical fields: x, y (1D), vx, vy (2D ny×nx), seeds_x, seeds_y, right_x.
    Optional: dt, max_steps, exit_penalty_time, y_policy, max_points (ignored), etc.

    Returns
    -------
    dict with keys particle_id, travel_time, x, y (path in physical coordinates).
    """
    _ensure_ti_init()

    x = np.asarray(params["x"], dtype=np.float64).ravel()
    y = np.asarray(params["y"], dtype=np.float64).ravel()
    vx = np.asarray(params["vx"], dtype=np.float64)
    vy = np.asarray(params["vy"], dtype=np.float64)
    seeds_x = np.asarray(params["seeds_x"], dtype=np.float32).ravel()
    seeds_y = np.asarray(params["seeds_y"], dtype=np.float32).ravel()
    right_x = float(params.get("right_x", x[-1]))

    x_unif, y_unif, vx_unif, vy_unif, xmin, xmax, ymin, ymax = _resample_velocity_to_uniform(x, y, vx, vy)
    vel_grid = np.stack([vx_unif, vy_unif], axis=2)

    dt = float(params.get("dt", 0.01))
    max_steps = int(params.get("max_steps", 5000))
    exit_penalty = float(params.get("exit_penalty_time", 1e9))
    y_policy = params.get("y_policy", "stop")
    closed = params.get("closed_boundary", True)

    sim = ParticleTracking2D(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        vel_grid=vel_grid,
        seeds_x=seeds_x,
        seeds_y=seeds_y,
        right_x=right_x,
        dt=dt,
        max_steps=max_steps,
        exit_penalty_time=exit_penalty,
        y_policy=y_policy,
        closed_boundary=closed,
        diffusion_sigma=0.0,
    )
    winner_idx, min_time, _ = sim.run_pass1()
    if min_time >= exit_penalty * 0.9:
        raise RuntimeError("No particles reached the target in rk4_fastest_path_params.")
    path_x, path_y = sim.replay_winner(winner_idx)
    return {"particle_id": winner_idx, "travel_time": min_time, "x": path_x, "y": path_y}


def rk4_fastest_path(vel_field: np.ndarray, cfg: SimConfig | None = None) -> dict:
    """
    Run the Taichi-based RK4 simulation on a normalized [0,1]×[0,1] velocity field.

    Parameters
    ----------
    vel_field : np.ndarray
        Velocity field on a unit square, shape (L, L, 2) with (vx, vy).
    cfg : SimConfig | None
        Configuration. If None, uses SimConfig() defaults.

    Returns
    -------
    dict with keys particle_id, travel_time, x, y (path in [0,1] coordinates).
    """
    if cfg is None:
        cfg = SimConfig()
    _ensure_ti_init()

    vel_field = np.asarray(vel_field, dtype=np.float32)
    if vel_field.ndim != 3 or vel_field.shape[2] != 2:
        raise ValueError("vel_field must have shape (L, L, 2).")
    L = vel_field.shape[0]
    n_particles = cfg.n_particles
    seeds_x = np.full(n_particles, cfg.source_x, dtype=np.float32)
    seeds_y = np.linspace(cfg.y_min, cfg.y_max, n_particles, dtype=np.float32)

    sim = ParticleTracking2D(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        vel_grid=vel_field,
        seeds_x=seeds_x,
        seeds_y=seeds_y,
        right_x=cfg.target_x,
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        exit_penalty_time=cfg.exit_penalty_time,
        y_policy="stop",
        closed_boundary=cfg.closed_boundary,
        diffusion_sigma=cfg.diffusion_sigma,
    )
    winner_idx, min_time, _ = sim.run_pass1()
    if min_time >= cfg.exit_penalty_time * 0.9:
        raise RuntimeError("No particles reached the target in rk4_fastest_path.")
    path_x, path_y = sim.replay_winner(winner_idx)
    return {"particle_id": winner_idx, "travel_time": min_time, "x": path_x, "y": path_y}
