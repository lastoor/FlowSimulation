"""Taichi backend for 2D PAR2-style random-walk particle tracking."""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np

STATUS_SUCCESS = np.int32(0)
STATUS_STAGNATION = np.int32(1)
STATUS_OUT_OF_BOUNDS = np.int32(2)
STATUS_MAX_STEPS = np.int32(4)
STATUS_SOURCE_INITIAL = np.int32(5)

PAR2_HUGE_TIME = np.float32(1.0e30)
PAR2_EPS = np.float32(1.0e-12)


@dataclass(frozen=True)
class Par2Inputs:
    x_faces: np.ndarray
    y_faces: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    theta_corners: np.ndarray
    seeds_x: np.ndarray
    seeds_y: np.ndarray
    dx: np.float32
    dy: np.float32
    dt: np.float32
    steps: int
    right_x: np.float32
    molecular_diffusion: np.float32
    alpha_l: np.float32
    alpha_t: np.float32
    seed: int
    max_points: int
    tracking_mode: str
    stop_mode: str
    arrival_fraction: float
    save_trajectories: bool
    return_fastest: bool
    snapshot_mode: str | None
    snapshot_times: object
    trajectory_memory_limit: int
    taichi_arch: str
    velocity_floor: np.float32


def require_uniform_faces(faces: np.ndarray, name: str) -> np.float32:
    """Validate a 1D uniform face-coordinate array and return its spacing."""
    faces = np.asarray(faces, dtype=np.float32)
    if faces.ndim != 1 or faces.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 values.")
    diffs = np.diff(faces).astype(np.float32)
    if np.any(diffs <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    if not np.allclose(diffs, diffs[0], rtol=1.0e-5, atol=1.0e-7):
        raise ValueError(
            f"{name} must be uniformly spaced for this PAR2 2D implementation."
        )
    return np.float32(diffs[0])


def compute_corner_velocities(
    vx_faces: np.ndarray,
    vy_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D corner velocities from staggered face velocities."""
    vx_faces = np.asarray(vx_faces, dtype=np.float32)
    vy_faces = np.asarray(vy_faces, dtype=np.float32)
    ny = vx_faces.shape[0]
    nx = vy_faces.shape[1]
    if vx_faces.shape != (ny, nx + 1):
        raise ValueError("vx_faces shape must be (ny, nx + 1).")
    if vy_faces.shape != (ny + 1, nx):
        raise ValueError("vy_faces shape must be (ny + 1, nx).")

    cvx = np.empty((ny + 1, nx + 1), dtype=np.float32)
    cvy = np.empty((ny + 1, nx + 1), dtype=np.float32)

    for iy in range(ny + 1):
        y_cells = [c for c in (iy - 1, iy) if 0 <= c < ny]
        for ix in range(nx + 1):
            x_cells = [c for c in (ix - 1, ix) if 0 <= c < nx]
            cvx[iy, ix] = np.mean([vx_faces[cell_y, ix] for cell_y in y_cells])
            cvy[iy, ix] = np.mean([vy_faces[iy, cell_x] for cell_x in x_cells])

    return cvx, cvy


def build_corner_porosity(theta, ny: int, nx: int) -> np.ndarray:
    """Normalize scalar, cell-centered, or corner porosity to corner values."""
    if theta is None:
        theta = 1.0
    arr = np.asarray(theta, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("porosity/theta must be positive and finite.")
        return np.full((ny + 1, nx + 1), value, dtype=np.float32)

    if arr.shape == (ny + 1, nx + 1):
        corners = arr.astype(np.float32, copy=True)
    elif arr.shape == (ny, nx):
        corners = np.empty((ny + 1, nx + 1), dtype=np.float32)
        for iy in range(ny + 1):
            y_cells = [c for c in (iy - 1, iy) if 0 <= c < ny]
            for ix in range(nx + 1):
                x_cells = [c for c in (ix - 1, ix) if 0 <= c < nx]
                corners[iy, ix] = np.mean(
                    [arr[cell_y, cell_x] for cell_y in y_cells for cell_x in x_cells]
                )
    else:
        raise ValueError(
            "porosity/theta must be scalar, shape (ny, nx), or shape "
            "(ny + 1, nx + 1)."
        )

    if np.any(~np.isfinite(corners)) or np.any(corners <= 0):
        raise ValueError("porosity/theta values must be positive and finite.")
    return corners.astype(np.float32)


def _regularized_velocity_2d(vx, vy, dm, alpha_l, velocity_floor=None):
    """Return robust velocity components for dispersion/eigenvector math."""
    vx = float(vx)
    vy = float(vy)
    dm = float(dm)
    alpha_l = float(alpha_l)
    if velocity_floor is None:
        velocity_floor = 0.01 * dm / alpha_l if alpha_l > 0.0 else 0.0
    velocity_floor = max(float(velocity_floor), 0.0)
    vnorm = math.hypot(vx, vy)
    if velocity_floor > 0.0 and vnorm < velocity_floor:
        return velocity_floor, 0.0, velocity_floor
    return vx, vy, vnorm


def dispersion_tensor_2d(vx, vy, dm, alpha_l, alpha_t, velocity_floor=None):
    """Compute the 2D block of the paper's local dispersion tensor."""
    vx, vy, vnorm = _regularized_velocity_2d(vx, vy, dm, alpha_l, velocity_floor)
    dm = float(dm)
    alpha_l = float(alpha_l)
    alpha_t = float(alpha_t)
    if vnorm <= float(PAR2_EPS):
        return np.array([[dm, 0.0], [0.0, dm]], dtype=np.float64)
    common = alpha_t * vnorm + dm
    beta = alpha_l - alpha_t
    return np.array(
        [
            [common + beta * vx * vx / vnorm, beta * vx * vy / vnorm],
            [beta * vx * vy / vnorm, common + beta * vy * vy / vnorm],
        ],
        dtype=np.float64,
    )


def analytic_displacement_matrix_2d(
    vx,
    vy,
    dm,
    alpha_l,
    alpha_t,
    dt=1.0,
    velocity_floor=None,
) -> np.ndarray:
    """Return the 2D analytic spectral displacement matrix.

    The returned matrix has shape ``(2, 2)`` and already includes ``sqrt(dt)``.
    Therefore ``B @ B.T`` equals ``2 * D * dt`` up to roundoff.
    """
    vx, vy, vnorm = _regularized_velocity_2d(vx, vy, dm, alpha_l, velocity_floor)
    dm = float(dm)
    alpha_l = float(alpha_l)
    alpha_t = float(alpha_t)
    dt = float(dt)
    if vnorm <= float(PAR2_EPS):
        coeff = math.sqrt(max(2.0 * dm * dt, 0.0))
        return np.array([[coeff, 0.0], [0.0, coeff]], dtype=np.float64)

    vnorm2 = vnorm * vnorm
    alpha = alpha_t * vnorm + dm
    beta = (alpha_l - alpha_t) / vnorm

    vectors = (
        (vx, vy, vnorm2, alpha + beta * vnorm2),
        (-vy, vx, vnorm2, alpha),
    )
    coeff = math.sqrt(max(2.0 * dt, 0.0))
    b = np.zeros((2, 2), dtype=np.float64)
    for ex, ey, norm2, gamma in vectors:
        if norm2 <= float(PAR2_EPS):
            continue
        scale = coeff * math.sqrt(max(gamma, 0.0)) / norm2
        b[0, 0] += scale * ex * ex
        b[0, 1] += scale * ex * ey
        b[1, 0] += scale * ey * ex
        b[1, 1] += scale * ey * ey
    return b


def make_store_steps(steps: int, max_points: int) -> np.ndarray:
    """Return monotonically increasing step ids to store for trajectories."""
    steps = int(steps)
    max_points = int(max_points)
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if steps < 1:
        return np.array([0], dtype=np.int32)
    if max_points == 1:
        return np.array([steps], dtype=np.int32)
    stride = max(1, int(math.ceil(steps / float(max_points - 1))))
    values = list(range(0, steps + 1, stride))
    if values[-1] != steps:
        values.append(steps)
    return np.asarray(values, dtype=np.int32)


def resolve_snapshot_times(snapshot_mode, raw_snapshot_times, travel_times):
    """Resolve reference-style snapshot params into absolute physical times."""
    if snapshot_mode is None or raw_snapshot_times is None:
        return None
    if snapshot_mode == "physical":
        return np.sort(np.asarray(raw_snapshot_times, dtype=np.float32).ravel())

    valid_times = np.asarray(travel_times, dtype=np.float32)
    valid_times = valid_times[np.isfinite(valid_times)]
    if valid_times.size == 0:
        return None

    if snapshot_mode == "dt_interval":
        dt = float(raw_snapshot_times)
        t_max = float(np.max(valid_times))
        if dt <= 0 or t_max <= 0:
            return None
        return np.arange(dt, t_max + 0.5 * dt, dt, dtype=np.float32)

    if snapshot_mode == "fraction_of_first_arrival":
        first = float(np.min(valid_times))
        if first <= 0:
            return None
        fractions = np.sort(np.asarray(raw_snapshot_times, dtype=np.float32).ravel())
        return (fractions * first).astype(np.float32)

    raise ValueError(
        "snapshot_mode must be 'physical', 'dt_interval', or "
        "'fraction_of_first_arrival'."
    )


def _first(params: dict, *names, default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


def _nested(params: dict, *path, default=None):
    node = params
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _physics_value(params: dict, canonical: str, yaml_name: str, short: str, default):
    physics = params.get("physics", {}) if isinstance(params.get("physics", {}), dict) else {}
    return _first(params, canonical, yaml_name, short, default=physics.get(yaml_name, default))


def _seeds_from_box(params: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_particles = _first(params, "n_particles", "particles")
    start_p1 = _first(params, "start_p1", "p1")
    start_p2 = _first(params, "start_p2", "p2")

    if n_particles is None:
        n_particles = _nested(params, "simulation", "particles", "N")
    if start_p1 is None:
        start_p1 = _nested(params, "simulation", "particles", "start", "p1")
    if start_p2 is None:
        start_p2 = _nested(params, "simulation", "particles", "start", "p2")

    if n_particles is None or start_p1 is None or start_p2 is None:
        raise ValueError(
            "Provide either seeds_x/seeds_y or n_particles with start_p1/start_p2."
        )

    p1 = np.asarray(start_p1, dtype=np.float32).ravel()
    p2 = np.asarray(start_p2, dtype=np.float32).ravel()
    if p1.size < 2 or p2.size < 2:
        raise ValueError("start_p1 and start_p2 must contain x and y coordinates.")
    if p2[0] < p1[0] or p2[1] < p1[1]:
        raise ValueError("start_p2 must be greater than or equal to start_p1.")

    rng = np.random.default_rng(int(seed))
    n_particles = int(n_particles)
    if n_particles < 1:
        raise ValueError("n_particles must be at least 1.")
    seeds_x = p1[0] + (p2[0] - p1[0]) * rng.random(n_particles, dtype=np.float32)
    seeds_y = p1[1] + (p2[1] - p1[1]) * rng.random(n_particles, dtype=np.float32)
    return seeds_x.astype(np.float32), seeds_y.astype(np.float32)


def parse_par2_inputs(params: dict) -> Par2Inputs:
    """Validate and normalize the user-facing configuration dict."""
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    x_faces = np.asarray(params["x_faces"], dtype=np.float32).ravel()
    y_faces = np.asarray(params["y_faces"], dtype=np.float32).ravel()
    dx = require_uniform_faces(x_faces, "x_faces")
    dy = require_uniform_faces(y_faces, "y_faces")

    vx = np.asarray(params["vx"], dtype=np.float32)
    vy = np.asarray(params["vy"], dtype=np.float32)
    nx = x_faces.size - 1
    ny = y_faces.size - 1
    if vx.shape != (ny, nx + 1):
        raise ValueError("vx shape must be (len(y_faces)-1, len(x_faces)).")
    if vy.shape != (ny + 1, nx):
        raise ValueError("vy shape must be (len(y_faces), len(x_faces)-1).")

    theta = _first(
        params,
        "theta",
        "porosity",
        default=_physics_value(params, "theta", "porosity", "rho", 1.0),
    )
    theta_corners = build_corner_porosity(theta, ny, nx)

    interpolation = str(params.get("interpolation", "trilinear")).lower()
    if interpolation not in ("trilinear", "bilinear"):
        raise NotImplementedError(
            "PyPar2 currently supports the PAR2 trilinear correction as a 2D "
            "bilinear corner-field correction only."
        )

    seed = int(params.get("seed", 0))
    if "seeds_x" in params or "seeds_y" in params:
        seeds_x = np.asarray(params["seeds_x"], dtype=np.float32).ravel()
        seeds_y = np.asarray(params["seeds_y"], dtype=np.float32).ravel()
        if seeds_x.shape != seeds_y.shape:
            raise ValueError("seeds_x and seeds_y must have the same length.")
        if seeds_x.size < 1:
            raise ValueError("at least one particle seed is required.")
    else:
        seeds_x, seeds_y = _seeds_from_box(params, seed)

    dt = _first(params, "dt", default=_nested(params, "simulation", "dt"))
    steps = _first(params, "steps", "max_steps", default=_nested(params, "simulation", "steps"))
    if dt is None or steps is None:
        raise ValueError("dt and steps are required.")
    dt = np.float32(dt)
    steps = int(steps)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    tracking_mode = str(params.get("tracking_mode", "arrival"))
    if tracking_mode not in ("arrival", "trajectory"):
        raise ValueError("tracking_mode must be 'arrival' or 'trajectory'.")
    stop_mode = str(params.get("stop_mode", "first"))
    if stop_mode not in ("first", "fraction", "all"):
        raise ValueError("stop_mode must be 'first', 'fraction', or 'all'.")
    taichi_arch = str(params.get("taichi_arch", "gpu")).lower()
    if taichi_arch not in ("gpu", "cpu"):
        raise ValueError("taichi_arch must be 'gpu' or 'cpu'.")

    save_trajectories = bool(params.get("save_trajectories", tracking_mode == "trajectory"))
    return_fastest = bool(params.get("return_fastest", stop_mode == "first"))
    molecular_diffusion = np.float32(
        _physics_value(params, "molecular_diffusion", "molecular diffusion", "Dm", 0.0)
    )
    alpha_l = np.float32(
        _physics_value(params, "longitudinal_dispersivity", "longitudinal dispersivity", "alphaL", 0.0)
    )
    alpha_t = np.float32(
        _physics_value(params, "transverse_dispersivity", "transverse dispersivity", "alphaT", 0.0)
    )
    if molecular_diffusion < 0:
        raise ValueError("molecular_diffusion must be nonnegative.")
    if alpha_l < 0 or alpha_t < 0:
        raise ValueError("dispersivities must be nonnegative.")
    if "velocity_floor" in params:
        velocity_floor = np.float32(params["velocity_floor"])
    else:
        velocity_floor = np.float32(
            0.01 * molecular_diffusion / alpha_l if alpha_l > 0 else 0.0
        )
    if velocity_floor < 0:
        raise ValueError("velocity_floor must be nonnegative.")

    return Par2Inputs(
        x_faces=x_faces,
        y_faces=y_faces,
        vx=vx,
        vy=vy,
        theta_corners=theta_corners,
        seeds_x=seeds_x.astype(np.float32),
        seeds_y=seeds_y.astype(np.float32),
        dx=dx,
        dy=dy,
        dt=dt,
        steps=steps,
        right_x=np.float32(params.get("right_x", x_faces[-1])),
        molecular_diffusion=molecular_diffusion,
        alpha_l=alpha_l,
        alpha_t=alpha_t,
        seed=seed,
        max_points=int(params.get("max_points", 2000)),
        tracking_mode=tracking_mode,
        stop_mode=stop_mode,
        arrival_fraction=float(params.get("arrival_fraction", 1.0)),
        save_trajectories=save_trajectories,
        return_fastest=return_fastest,
        snapshot_mode=params.get("snapshot_mode", None),
        snapshot_times=params.get("snapshot_times", None),
        trajectory_memory_limit=int(params.get("trajectory_memory_limit", 2_000_000)),
        taichi_arch=taichi_arch,
        velocity_floor=velocity_floor,
    )

_ti_initialized = False
_ti_arch = None
_tracker_cls = None


def _probe_taichi_runtime(ti):
    """Force a tiny host/device transfer so unusable GPU runtimes fail early."""
    probe = ti.field(dtype=ti.f32, shape=1)
    probe.from_numpy(np.asarray([1.0], dtype=np.float32))
    _ = probe.to_numpy()


def _ensure_ti_init(preferred_arch="gpu"):
    global _ti_initialized, _ti_arch
    if _ti_initialized:
        return _ti_arch
    import taichi as ti

    # If another FlowSimulation taichi-using module (e.g. pollock_gpu) has
    # already initialized Taichi in this process, re-running ti.init would
    # reset the global runtime and wipe any persistent fields allocated by
    # those other modules. Detect that via a shared sentinel on the taichi
    # module and a best-effort check on pollock_gpu's own init flag.
    external_arch = None
    if bool(getattr(ti, "_fs_taichi_initialized", False)):
        external_arch = getattr(ti, "_fs_taichi_arch", None)
    if external_arch is None:
        try:
            from . import pollock_gpu as _pg
            if bool(getattr(_pg, "_ti_initialized", False)):
                external_arch = getattr(_pg, "_ti_arch", None) or "gpu"
        except Exception:
            external_arch = None
    if external_arch is not None:
        _ti_initialized = True
        _ti_arch = external_arch
        setattr(ti, "_fs_taichi_initialized", True)
        setattr(ti, "_fs_taichi_arch", _ti_arch)
        return _ti_arch

    cache_path = os.environ.get(
        "PYPAR2_TAICHI_CACHE",
        os.path.join(tempfile.gettempdir(), "pypar2_ticache"),
    )
    os.makedirs(cache_path, exist_ok=True)
    init_kwargs = {
        "verbose": False,
        "offline_cache": True,
        "offline_cache_file_path": cache_path,
    }
    if preferred_arch == "cpu":
        ti.init(arch=ti.cpu, **init_kwargs)
        _probe_taichi_runtime(ti)
        _ti_initialized = True
        _ti_arch = "cpu"
        setattr(ti, "_fs_taichi_initialized", True)
        setattr(ti, "_fs_taichi_arch", _ti_arch)
        return _ti_arch

    try:
        ti.init(arch=ti.gpu, **init_kwargs)
        _probe_taichi_runtime(ti)
        _ti_arch = "gpu"
    except Exception as exc:
        try:
            ti.reset()
        except Exception:
            pass
        warnings.warn(
            f"Taichi GPU initialization failed; falling back to CPU ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        ti.init(arch=ti.cpu, **init_kwargs)
        _probe_taichi_runtime(ti)
        _ti_arch = "cpu"
    _ti_initialized = True
    setattr(ti, "_fs_taichi_initialized", True)
    setattr(ti, "_fs_taichi_arch", _ti_arch)
    return _ti_arch


def _initial_rng_states(seed: int, n_particles: int) -> np.ndarray:
    ids = np.arange(n_particles, dtype=np.uint64) + np.uint64(1)
    state = (np.uint64(seed) ^ (ids * np.uint64(0x9E3779B9))) & np.uint64(0xFFFFFFFF)
    state = np.where(state == 0, np.uint64(0xA341316C), state)
    return state.astype(np.uint32)


def _make_tracker_cls():
    global _tracker_cls
    if _tracker_cls is not None:
        return _tracker_cls

    import taichi as ti

    @ti.data_oriented
    class Par2RWPT2D:
        def __init__(
            self,
            cfg: Par2Inputs,
            snapshot_times=None,
            store_trajectories=False,
        ):
            self.nx = cfg.x_faces.size - 1
            self.ny = cfg.y_faces.size - 1
            self.n_particles = cfg.seeds_x.size
            self.steps = int(cfg.steps)
            self.dt = float(cfg.dt)
            self.dx = float(cfg.dx)
            self.dy = float(cfg.dy)
            self.x_min = float(cfg.x_faces[0])
            self.x_max = float(cfg.x_faces[-1])
            self.y_min = float(cfg.y_faces[0])
            self.y_max = float(cfg.y_faces[-1])
            self.right_x = float(cfg.right_x)
            self.dm = float(cfg.molecular_diffusion)
            self.alpha_l = float(cfg.alpha_l)
            self.alpha_t = float(cfg.alpha_t)
            self.seed = int(cfg.seed)
            self.velocity_floor = float(cfg.velocity_floor)

            cvx, cvy = compute_corner_velocities(cfg.vx, cfg.vy)
            self.x_faces = ti.field(dtype=ti.f32, shape=cfg.x_faces.size)
            self.y_faces = ti.field(dtype=ti.f32, shape=cfg.y_faces.size)
            self.vx_faces = ti.field(dtype=ti.f32, shape=cfg.vx.shape)
            self.vy_faces = ti.field(dtype=ti.f32, shape=cfg.vy.shape)
            self.cvx = ti.field(dtype=ti.f32, shape=cvx.shape)
            self.cvy = ti.field(dtype=ti.f32, shape=cvy.shape)
            self.theta_corners = ti.field(dtype=ti.f32, shape=cfg.theta_corners.shape)
            self.x_faces.from_numpy(cfg.x_faces)
            self.y_faces.from_numpy(cfg.y_faces)
            self.vx_faces.from_numpy(cfg.vx)
            self.vy_faces.from_numpy(cfg.vy)
            self.cvx.from_numpy(cvx)
            self.cvy.from_numpy(cvy)
            self.theta_corners.from_numpy(cfg.theta_corners)

            self.seed_x = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.seed_y = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.pos_x = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.pos_y = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.elapsed = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.final_times = ti.field(dtype=ti.f32, shape=self.n_particles)
            self.status = ti.field(dtype=ti.i32, shape=self.n_particles)
            self.active = ti.field(dtype=ti.i32, shape=self.n_particles)
            self.rng_state = ti.field(dtype=ti.u32, shape=self.n_particles)
            self.seed_x.from_numpy(cfg.seeds_x)
            self.seed_y.from_numpy(cfg.seeds_y)
            self.rng_state.from_numpy(_initial_rng_states(cfg.seed, self.n_particles))

            if snapshot_times is None:
                snap_np = np.empty((0,), dtype=np.float32)
            else:
                snap_np = np.sort(np.asarray(snapshot_times, dtype=np.float32).ravel())
            self.n_snaps = int(snap_np.size)
            self.snap_times = ti.field(dtype=ti.f32, shape=max(self.n_snaps, 1))
            self.snap_x = ti.field(dtype=ti.f32, shape=(self.n_particles, max(self.n_snaps, 1)))
            self.snap_y = ti.field(dtype=ti.f32, shape=(self.n_particles, max(self.n_snaps, 1)))
            if self.n_snaps > 0:
                self.snap_times.from_numpy(snap_np)

            self.store_trajectories = bool(store_trajectories)
            if self.store_trajectories:
                store_steps = make_store_steps(self.steps, cfg.max_points)
            else:
                store_steps = np.array([0], dtype=np.int32)
            self.n_traj = int(store_steps.size)
            self.traj_steps = ti.field(dtype=ti.i32, shape=self.n_traj)
            self.traj_x = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_traj))
            self.traj_y = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_traj))
            self.traj_t = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_traj))
            self.traj_steps.from_numpy(store_steps)

        def _reset_output_fields(self):
            if self.n_snaps > 0:
                empty_snaps = np.full(
                    (self.n_particles, self.n_snaps), np.nan, dtype=np.float32
                )
                self.snap_x.from_numpy(empty_snaps)
                self.snap_y.from_numpy(empty_snaps)
            if self.store_trajectories:
                empty_traj = np.full(
                    (self.n_particles, self.n_traj), np.nan, dtype=np.float32
                )
                self.traj_x.from_numpy(empty_traj)
                self.traj_y.from_numpy(empty_traj)
                self.traj_t.from_numpy(empty_traj)

        @ti.func
        def _linear(self, x, v0, v1):
            return v0 * (1.0 - x) + v1 * x

        @ti.func
        def _bilinear(self, x, y, v00, v10, v01, v11):
            v0 = self._linear(x, v00, v10)
            v1 = self._linear(x, v01, v11)
            return self._linear(y, v0, v1)

        @ti.func
        def _valid_x(self, x):
            return x > self.x_min and x < self.x_max

        @ti.func
        def _valid_y(self, y):
            return y > self.y_min and y < self.y_max

        @ti.func
        def _cell_j(self, x):
            j = -1
            if self._valid_x(x):
                j = ti.cast(ti.floor((x - self.x_min) / self.dx), ti.i32)
            return j

        @ti.func
        def _cell_i(self, y):
            i = -1
            if self._valid_y(y):
                i = ti.cast(ti.floor((y - self.y_min) / self.dy), ti.i32)
            return i

        @ti.func
        def _face_velocity(self, i, j, x, y):
            lx = (x - self.x_faces[j]) / self.dx
            ly = (y - self.y_faces[i]) / self.dy
            vx = self._linear(lx, self.vx_faces[i, j], self.vx_faces[i, j + 1])
            vy = self._linear(ly, self.vy_faces[i, j], self.vy_faces[i + 1, j])
            return vx, vy, lx, ly

        @ti.func
        def _regularized_velocity(self, vx, vy):
            vnorm = ti.sqrt(vx * vx + vy * vy)
            out_vx = vx
            out_vy = vy
            out_norm = vnorm
            if self.velocity_floor > 0.0 and vnorm < self.velocity_floor:
                out_vx = self.velocity_floor
                out_vy = 0.0
                out_norm = self.velocity_floor
            return out_vx, out_vy, out_norm

        @ti.func
        def _dispersion_components(self, vx, vy):
            vx, vy, vnorm = self._regularized_velocity(vx, vy)
            dxx = self.dm
            dyy = self.dm
            dxy = 0.0
            if vnorm > PAR2_EPS:
                common = self.alpha_t * vnorm + self.dm
                beta = self.alpha_l - self.alpha_t
                dxx = common + beta * vx * vx / vnorm
                dyy = common + beta * vy * vy / vnorm
                dxy = beta * vx * vy / vnorm
            return dxx, dyy, dxy

        @ti.func
        def _corner_velocity(self, i, j, lx, ly):
            vx = self._bilinear(
                lx,
                ly,
                self.cvx[i, j],
                self.cvx[i, j + 1],
                self.cvx[i + 1, j],
                self.cvx[i + 1, j + 1],
            )
            vy = self._bilinear(
                lx,
                ly,
                self.cvy[i, j],
                self.cvy[i, j + 1],
                self.cvy[i + 1, j],
                self.cvy[i + 1, j + 1],
            )
            return vx, vy

        @ti.func
        def _theta_and_gradient(self, i, j, lx, ly):
            theta00 = self.theta_corners[i, j]
            theta10 = self.theta_corners[i, j + 1]
            theta01 = self.theta_corners[i + 1, j]
            theta11 = self.theta_corners[i + 1, j + 1]
            theta = self._bilinear(lx, ly, theta00, theta10, theta01, theta11)
            dtheta_dx = self._linear(
                ly, (theta10 - theta00) / self.dx, (theta11 - theta01) / self.dx
            )
            dtheta_dy = self._linear(
                lx, (theta01 - theta00) / self.dy, (theta11 - theta10) / self.dy
            )
            return theta, dtheta_dx, dtheta_dy

        @ti.func
        def _porosity_drift(self, i, j, lx, ly):
            theta, dtheta_dx, dtheta_dy = self._theta_and_gradient(i, j, lx, ly)
            vx, vy = self._corner_velocity(i, j, lx, ly)
            dxx, dyy, dxy = self._dispersion_components(vx, vy)
            safe_theta = ti.max(theta, PAR2_EPS)
            return (
                (dxx * dtheta_dx + dxy * dtheta_dy) / safe_theta,
                (dxy * dtheta_dx + dyy * dtheta_dy) / safe_theta,
            )

        @ti.func
        def _drift_correction(self, i, j, lx, ly):
            dxx00, dyy00, dxy00 = self._dispersion_components(self.cvx[i, j], self.cvy[i, j])
            dxx10, dyy10, dxy10 = self._dispersion_components(self.cvx[i, j + 1], self.cvy[i, j + 1])
            dxx01, dyy01, dxy01 = self._dispersion_components(self.cvx[i + 1, j], self.cvy[i + 1, j])
            dxx11, dyy11, dxy11 = self._dispersion_components(self.cvx[i + 1, j + 1], self.cvy[i + 1, j + 1])

            d_dxx_dx = self._linear(ly, (dxx10 - dxx00) / self.dx, (dxx11 - dxx01) / self.dx)
            d_dxy_dy = self._linear(lx, (dxy01 - dxy00) / self.dy, (dxy11 - dxy10) / self.dy)
            d_dxy_dx = self._linear(ly, (dxy10 - dxy00) / self.dx, (dxy11 - dxy01) / self.dx)
            d_dyy_dy = self._linear(lx, (dyy01 - dyy00) / self.dy, (dyy11 - dyy10) / self.dy)
            return d_dxx_dx + d_dxy_dy, d_dxy_dx + d_dyy_dy

        @ti.func
        def _random_uniform(self, p):
            state = self.rng_state[p]
            state = state * ti.u32(1664525) + ti.u32(1013904223)
            self.rng_state[p] = state
            return (ti.cast(state, ti.f32) + 0.5) * 2.3283064365386963e-10

        @ti.func
        def _random_normals(self, p):
            u1 = self._random_uniform(p)
            u2 = self._random_uniform(p)
            if u1 < 1.0e-7:
                u1 = 1.0e-7
            radius = ti.sqrt(-2.0 * ti.log(u1))
            theta = 6.283185307179586 * u2
            return radius * ti.cos(theta), radius * ti.sin(theta)

        @ti.func
        def _displacement_matrix(self, i, j, lx, ly):
            vx, vy = self._corner_velocity(i, j, lx, ly)
            vx, vy, vnorm = self._regularized_velocity(vx, vy)

            b00 = ti.sqrt(ti.max(2.0 * self.dm * self.dt, 0.0))
            b11 = b00
            b01 = 0.0

            if vnorm > PAR2_EPS:
                vnorm2 = vnorm * vnorm
                alpha = self.alpha_t * vnorm + self.dm
                beta = (self.alpha_l - self.alpha_t) / vnorm

                vx1 = -vy
                vy1 = vx

                vnorm2_1 = vx1 * vx1 + vy1 * vy1

                gamma0 = ti.sqrt(ti.max(alpha + beta * vnorm2, 0.0)) / vnorm2
                gamma1 = 0.0
                if vnorm2_1 > PAR2_EPS:
                    gamma1 = ti.sqrt(ti.max(alpha, 0.0)) / vnorm2_1

                coeff = ti.sqrt(2.0 * self.dt)
                b00 = coeff * (
                    gamma0 * vx * vx + gamma1 * vx1 * vx1
                )
                b11 = coeff * (
                    gamma0 * vy * vy + gamma1 * vy1 * vy1
                )
                b01 = coeff * (
                    gamma0 * vx * vy + gamma1 * vx1 * vy1
                )
            return b00, b11, b01

        @ti.func
        def _stochastic_displacement(self, p, i, j, lx, ly):
            b00, b11, b01 = self._displacement_matrix(i, j, lx, ly)
            xi0, xi1 = self._random_normals(p)
            return b00 * xi0 + b01 * xi1, b01 * xi0 + b11 * xi1

        @ti.func
        def _capture_snapshots(self, p, snap_idx, prev_x, prev_y, prev_t, curr_x, curr_y, curr_t):
            idx = snap_idx
            while idx < self.n_snaps and self.snap_times[idx] <= curr_t:
                snap_t = self.snap_times[idx]
                if snap_t >= prev_t:
                    alpha = (snap_t - prev_t) / (curr_t - prev_t + 1.0e-30)
                    self.snap_x[p, idx] = prev_x + alpha * (curr_x - prev_x)
                    self.snap_y[p, idx] = prev_y + alpha * (curr_y - prev_y)
                idx += 1
            return idx

        @ti.kernel
        def init_state(self):
            for p in range(self.n_particles):
                x = self.seed_x[p]
                y = self.seed_y[p]
                self.pos_x[p] = x
                self.pos_y[p] = y
                self.elapsed[p] = 0.0
                self.final_times[p] = PAR2_HUGE_TIME
                self.status[p] = STATUS_MAX_STEPS
                self.active[p] = 1

                if not (self._valid_x(x) and self._valid_y(y)):
                    self.status[p] = STATUS_OUT_OF_BOUNDS
                    self.final_times[p] = 0.0
                    self.active[p] = 0
                elif x >= self.right_x:
                    self.status[p] = STATUS_SUCCESS
                    self.final_times[p] = 0.0
                    self.active[p] = 0

                if self.store_trajectories:
                    self.traj_x[p, 0] = x
                    self.traj_y[p, 0] = y
                    self.traj_t[p, 0] = 0.0

                k = 0
                while k < self.n_snaps and self.snap_times[k] <= 0.0:
                    self.snap_x[p, k] = x
                    self.snap_y[p, k] = y
                    k += 1

        @ti.kernel
        def track_all(self):
            for p in range(self.n_particles):
                snap_idx = 0
                while snap_idx < self.n_snaps and self.snap_times[snap_idx] <= 0.0:
                    snap_idx += 1
                traj_idx = 1

                for step in range(self.steps):
                    wrote_step = 0
                    if self.active[p] == 1:
                        prev_x = self.pos_x[p]
                        prev_y = self.pos_y[p]
                        prev_t = self.elapsed[p]
                        i = self._cell_i(prev_y)
                        j = self._cell_j(prev_x)
                        if i < 0 or j < 0:
                            self.status[p] = STATUS_OUT_OF_BOUNDS
                            self.final_times[p] = prev_t
                            self.active[p] = 0
                        else:
                            vlx, vly, lx, ly = self._face_velocity(i, j, prev_x, prev_y)
                            vcx, vcy = self._drift_correction(i, j, lx, ly)
                            pcx, pcy = self._porosity_drift(i, j, lx, ly)
                            sdx, sdy = self._stochastic_displacement(p, i, j, lx, ly)

                            raw_x = prev_x + (vlx + vcx + pcx) * self.dt + sdx
                            raw_y = prev_y + (vly + vcy + pcy) * self.dt + sdy

                            curr_x = prev_x
                            curr_y = prev_y
                            if raw_x >= self.right_x:
                                curr_x = self.right_x
                                if self._valid_y(raw_y):
                                    curr_y = raw_y
                            else:
                                if self._valid_x(raw_x):
                                    curr_x = raw_x
                                if self._valid_y(raw_y):
                                    curr_y = raw_y

                            curr_t = prev_t + self.dt
                            if self.n_snaps > 0:
                                snap_idx = self._capture_snapshots(
                                    p, snap_idx, prev_x, prev_y, prev_t, curr_x, curr_y, curr_t
                                )

                            self.pos_x[p] = curr_x
                            self.pos_y[p] = curr_y
                            self.elapsed[p] = curr_t
                            wrote_step = 1

                            if curr_x >= self.right_x:
                                self.status[p] = STATUS_SUCCESS
                                self.final_times[p] = curr_t
                                self.active[p] = 0

                    if self.store_trajectories:
                        while wrote_step == 1 and traj_idx < self.n_traj and self.traj_steps[traj_idx] <= step + 1:
                            self.traj_x[p, traj_idx] = self.pos_x[p]
                            self.traj_y[p, traj_idx] = self.pos_y[p]
                            self.traj_t[p, traj_idx] = self.elapsed[p]
                            traj_idx += 1

                if self.active[p] == 1:
                    self.status[p] = STATUS_MAX_STEPS
                    self.final_times[p] = self.elapsed[p]
                    self.active[p] = 0

        def run(self):
            self._reset_output_fields()
            self.rng_state.from_numpy(_initial_rng_states(self.seed, self.n_particles))
            self.init_state()
            self.track_all()
            ti.sync()
            result = {
                "final_x": self.pos_x.to_numpy(),
                "final_y": self.pos_y.to_numpy(),
                "raw_times": self.final_times.to_numpy(),
                "status": self.status.to_numpy(),
            }
            if self.n_snaps > 0:
                result["snap_x"] = self.snap_x.to_numpy()[:, : self.n_snaps]
                result["snap_y"] = self.snap_y.to_numpy()[:, : self.n_snaps]
                result["snapshot_times"] = self.snap_times.to_numpy()[: self.n_snaps]
            else:
                result["snap_x"] = None
                result["snap_y"] = None
                result["snapshot_times"] = None
            if self.store_trajectories:
                result["traj_x"] = self.traj_x.to_numpy()
                result["traj_y"] = self.traj_y.to_numpy()
                result["traj_t"] = self.traj_t.to_numpy()
            else:
                result["traj_x"] = None
                result["traj_y"] = None
                result["traj_t"] = None
            return result

    _tracker_cls = Par2RWPT2D
    return _tracker_cls


def _successful_times(raw_times, status):
    return np.where(status == STATUS_SUCCESS, raw_times, np.nan).astype(np.float32)


def _selected_particle_ids(travel_times, stop_mode, arrival_fraction):
    arrived = np.flatnonzero(np.isfinite(travel_times))
    if arrived.size == 0:
        return np.empty((0,), dtype=np.int32)
    order = arrived[np.argsort(travel_times[arrived])]
    if stop_mode == "first":
        return order[:1].astype(np.int32)
    if stop_mode == "fraction":
        fraction = float(arrival_fraction)
        if fraction <= 0:
            return np.empty((0,), dtype=np.int32)
        if fraction <= 1.0:
            count = max(1, int(np.ceil(fraction * order.size)))
            return order[:count].astype(np.int32)
        threshold = np.nanmin(travel_times) * fraction
        return order[travel_times[order] <= threshold].astype(np.int32)
    return order.astype(np.int32)


def _build_summary(cfg, result, taichi_arch, resolved_snapshots, trajectories_enabled):
    status = result["status"].astype(np.int32)
    raw_times = result["raw_times"].astype(np.float32)
    travel_times = _successful_times(raw_times, status)
    selected_ids = _selected_particle_ids(
        travel_times, cfg.stop_mode, cfg.arrival_fraction
    )
    arrived = np.isfinite(travel_times)

    summary = {
        "backend": "gpu",
        "taichi_arch": taichi_arch,
        "method": "par2",
        "tracking_mode": cfg.tracking_mode,
        "stop_mode": cfg.stop_mode,
        "save_trajectories": cfg.save_trajectories,
        "return_fastest": cfg.return_fastest,
        "status": status,
        "lengths": None,
        "travel_times": travel_times,
        "raw_travel_times": raw_times,
        "final_x": result["final_x"].astype(np.float32),
        "final_y": result["final_y"].astype(np.float32),
        "n_particles": int(status.size),
        "n_arrived": int(np.count_nonzero(arrived)),
        "arrival_fraction": float(np.count_nonzero(arrived) / max(status.size, 1)),
        "selected_particle_ids": selected_ids,
        "concentration_after_right_x": float(np.count_nonzero(arrived) / max(status.size, 1)),
        "trajectory_storage": "enabled" if trajectories_enabled else "disabled",
    }

    if summary["n_arrived"] > 0:
        summary["first_arrival_time"] = np.float32(np.nanmin(travel_times))
        summary["last_arrival_time"] = np.float32(np.nanmax(travel_times))
    else:
        summary["first_arrival_time"] = np.nan
        summary["last_arrival_time"] = np.nan

    if resolved_snapshots is not None and result["snap_x"] is not None:
        summary["snapshots"] = {
            "snapshot_mode": cfg.snapshot_mode,
            "snapshot_times": resolved_snapshots.astype(np.float32),
            "x": result["snap_x"].astype(np.float32),
            "y": result["snap_y"].astype(np.float32),
        }

    selected_trajectories = []
    if trajectories_enabled and result["traj_x"] is not None:
        for particle_id in selected_ids:
            pid = int(particle_id)
            valid = np.isfinite(result["traj_t"][pid])
            selected_trajectories.append(
                {
                    "particle_id": pid,
                    "x": result["traj_x"][pid, valid].astype(np.float32).copy(),
                    "y": result["traj_y"][pid, valid].astype(np.float32).copy(),
                    "t": result["traj_t"][pid, valid].astype(np.float32).copy(),
                }
            )
    elif cfg.save_trajectories:
        summary["trajectory_warning"] = (
            "Trajectory storage was disabled because n_particles * max_points "
            "exceeded trajectory_memory_limit."
        )
    if cfg.save_trajectories:
        summary["selected_trajectories"] = selected_trajectories

    if cfg.return_fastest and summary["n_arrived"] > 0:
        fastest_id = int(np.nanargmin(travel_times))
        fastest = {
            "particle_id": fastest_id,
            "travel_time": np.float32(travel_times[fastest_id]),
            "x": np.asarray([result["final_x"][fastest_id]], dtype=np.float32),
            "y": np.asarray([result["final_y"][fastest_id]], dtype=np.float32),
            "t": np.asarray([travel_times[fastest_id]], dtype=np.float32),
        }
        if trajectories_enabled and result["traj_x"] is not None:
            valid = np.isfinite(result["traj_t"][fastest_id])
            fastest.update(
                {
                    "x": result["traj_x"][fastest_id, valid].astype(np.float32).copy(),
                    "y": result["traj_y"][fastest_id, valid].astype(np.float32).copy(),
                    "t": result["traj_t"][fastest_id, valid].astype(np.float32).copy(),
                }
            )
        summary["fastest"] = fastest

    return summary


def _run_once(cfg: Par2Inputs, snapshot_times=None, force_trajectories=False):
    taichi_arch = _ensure_ti_init(cfg.taichi_arch)
    Tracker = _make_tracker_cls()
    store_trajectories = bool(
        force_trajectories
        and cfg.seeds_x.size * max(1, cfg.max_points) <= cfg.trajectory_memory_limit
    )
    tracker = Tracker(cfg, snapshot_times=snapshot_times, store_trajectories=store_trajectories)
    return tracker.run(), taichi_arch, store_trajectories


def particle_tracking_par2_gpu_summary(params):
    """Run 2D PAR2-style RWPT and return a reference-shaped summary dict."""
    cfg = parse_par2_inputs(params)

    resolved_snapshots = None
    if cfg.snapshot_mode is not None and cfg.snapshot_times is not None:
        if cfg.snapshot_mode == "physical":
            resolved_snapshots = resolve_snapshot_times(
                cfg.snapshot_mode, cfg.snapshot_times, None
            )
        else:
            pass1, _, _ = _run_once(cfg, snapshot_times=None, force_trajectories=False)
            resolved_snapshots = resolve_snapshot_times(
                cfg.snapshot_mode,
                cfg.snapshot_times,
                _successful_times(pass1["raw_times"], pass1["status"]),
            )

    force_trajectories = cfg.save_trajectories or cfg.tracking_mode == "trajectory" or cfg.return_fastest
    result, taichi_arch, trajectories_enabled = _run_once(
        cfg, snapshot_times=resolved_snapshots, force_trajectories=force_trajectories
    )
    return _build_summary(cfg, result, taichi_arch, resolved_snapshots, trajectories_enabled)


def particle_tracking_pollock_gpu_summary(params):
    """Compatibility alias using PAR2 RWPT physics instead of Pollock pathlines."""
    return particle_tracking_par2_gpu_summary(params)


def pollock_source_arrival_times_gpu(*_args, **_kwargs):
    """Source-mask Pollock tracking is not part of the PAR2 RWPT port."""
    raise NotImplementedError(
        "pollock_source_arrival_times_gpu is not implemented for PyPar2's "
        "PAR2 random-walk backend."
    )
