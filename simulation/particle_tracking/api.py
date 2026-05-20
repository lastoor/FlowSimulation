"""High-level particle-tracking dispatch helpers."""

import numpy as np

from .pollock_cpu import particle_tracking_pollock_summary
from .pollock_gpu import particle_tracking_pollock_gpu_summary
from .rk45_cpu import particle_tracking_cpu_summary
from .utils import (
    build_fastest_result_from_arrays,
    extract_particle_trajectories,
    summarize_arrivals,
)


def particle_tracking_rk45_gpu_summary(params):
    """High-level GPU RK45 tracking API with summary shaping."""
    try:
        from .rk45_gpu import gpu_track_particles_arclength
    except Exception as exc:
        raise RuntimeError("RK45 GPU backend is unavailable in this environment.") from exc

    tracking_mode = params.get("tracking_mode", "trajectory")
    if tracking_mode not in ("trajectory", "arrival"):
        raise ValueError("tracking_mode must be 'trajectory' or 'arrival'.")

    stop_mode = params.get("stop_mode", "first")
    arrival_fraction = float(params.get("arrival_fraction", 1.0))
    save_trajectories = bool(params.get("save_trajectories", tracking_mode == "trajectory"))
    return_fastest = bool(params.get("return_fastest", stop_mode == "first"))

    px, py, pt, L, st, _, _, _ = gpu_track_particles_arclength(
        params["x"],
        params["y"],
        params["vx"],
        params["vy"],
        params["seeds_x"],
        params["seeds_y"],
        right_x=params.get("right_x"),
        eps_speed=params.get("eps_speed", 1e-16),
        ds_min=params.get("ds_min", 1e-5),
        ds_max=params.get("ds_max", 1.0),
        tol=params.get("tol", 1e-4),
        y_policy=params.get("y_policy", "stop"),
        max_points=params.get("max_points", 2000),
        max_iters=params.get("max_iters", 200000),
        threads_per_block=params.get("threads_per_block", 256),
        dtype=params.get("dtype", np.float32),
        velocity_location=params.get("velocity_location", "cell"),
        x_cell_centers=params.get("x_cell_centers"),
        y_cell_centers=params.get("y_cell_centers"),
    )

    travel_times = np.full(L.shape[0], np.nan, dtype=pt.dtype)
    valid = (st == 0) & (L > 0)
    idx = np.where(valid)[0]
    if idx.size > 0:
        travel_times[idx] = pt[idx, L[idx] - 1]

    summary = summarize_arrivals(travel_times, st, stop_mode=stop_mode, arrival_fraction=arrival_fraction)
    summary.update(
        {
            "backend": "gpu",
            "method": "rk45",
            "tracking_mode": tracking_mode,
            "stop_mode": stop_mode,
            "save_trajectories": save_trajectories,
            "return_fastest": return_fastest,
            "status": st,
            "lengths": L,
        }
    )

    if tracking_mode == "trajectory" or save_trajectories:
        summary["px"] = px
        summary["py"] = py
        summary["pt"] = pt

    if save_trajectories:
        summary["selected_trajectories"] = extract_particle_trajectories(px, py, pt, L, summary["selected_particle_ids"])

    if return_fastest:
        summary["fastest"] = build_fastest_result_from_arrays(px, py, pt, L, travel_times)

    return summary


def _resolve_method_and_backend(params):
    backend = params.get("backend", "cpu")
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'.")

    method = params.get("method")
    if method is None:
        method = "rk45" if backend == "cpu" else "rk4"
    if method not in ("pollock", "rk45", "rk4", "pypar2"):
        raise ValueError("method must be 'pollock', 'rk45', 'rk4', or 'pypar2'.")
    return method, backend


def particle_tracking_summary(params):
    """
    Dispatch particle tracking to CPU RK45 or GPU Taichi backends.

    Existing low-level tuple-returning APIs remain unchanged; this function
    provides the new arrival/trajectory summary interface.
    """
    method, backend = _resolve_method_and_backend(params)

    if method == "pollock":
        if backend == "cpu":
            return particle_tracking_pollock_summary(params)
        if backend == "gpu":
            return particle_tracking_pollock_gpu_summary(params)

    if method == "rk45":
        if backend == "cpu":
            return particle_tracking_cpu_summary(params)
        if backend == "gpu":
            return particle_tracking_rk45_gpu_summary(params)

    if method == "rk4":
        if backend != "gpu":
            raise ValueError("method='rk4' currently supports backend='gpu' only.")
        from .rk4_fp import rk4_particle_tracking_summary
        return rk4_particle_tracking_summary(params)

    if method == "pypar2":
        from .pypar2 import particle_tracking_par2_gpu_summary
        forwarded = dict(params)
        forwarded.setdefault("taichi_arch", backend)
        return particle_tracking_par2_gpu_summary(forwarded)

    if backend == "cpu":
        return particle_tracking_cpu_summary(params)

    raise ValueError(f"Unsupported method/backend combination: method={method!r}, backend={backend!r}.")
