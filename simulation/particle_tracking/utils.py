"""
Particle-tracking utilities: snapshot-at-t0 for plume visualization.
"""

import numpy as np


def compute_travel_times(pt, L, st):
    """Return per-particle travel times, NaN for unsuccessful trajectories."""
    nP = L.shape[0]
    travel_times = np.full(nP, np.nan, dtype=pt.dtype)
    valid = (st == 0) & (L > 0)
    idx = np.where(valid)[0]
    if idx.size > 0:
        travel_times[idx] = pt[idx, L[idx] - 1]
    return travel_times


def summarize_arrivals(travel_times, status, stop_mode="first", arrival_fraction=1.0):
    """Summarize arrival order and stopping particle selection from per-particle travel times."""
    travel_times = np.asarray(travel_times, dtype=np.float64)
    status = np.asarray(status, dtype=np.int32)

    if stop_mode not in ("first", "fraction", "full_btc"):
        raise ValueError("stop_mode must be 'first', 'fraction', or 'full_btc'.")

    if not (0.0 < float(arrival_fraction) <= 1.0):
        raise ValueError("arrival_fraction must lie in (0, 1].")

    valid = (status == 0) & np.isfinite(travel_times)
    arrival_ids = np.flatnonzero(valid).astype(np.int32)
    order = np.argsort(travel_times[arrival_ids], kind="stable") if arrival_ids.size else np.array([], dtype=np.int64)
    arrival_order_ids = arrival_ids[order]
    arrival_order_times = travel_times[arrival_order_ids]

    n_particles = int(status.size)
    if stop_mode == "first":
        target_count = 1
    elif stop_mode == "fraction":
        target_count = int(np.ceil(float(arrival_fraction) * n_particles))
    else:
        target_count = int(arrival_order_ids.size)

    threshold_reached = arrival_order_ids.size >= target_count and target_count > 0
    if threshold_reached:
        selected_particle_ids = arrival_order_ids[:target_count].copy()
        selected_travel_times = arrival_order_times[:target_count].copy()
        stopping_particle_id = int(selected_particle_ids[-1])
        stopping_travel_time = float(selected_travel_times[-1])
    else:
        selected_particle_ids = arrival_order_ids.copy()
        selected_travel_times = arrival_order_times.copy()
        stopping_particle_id = None
        stopping_travel_time = np.nan

    return {
        "travel_times": travel_times,
        "arrival_order_ids": arrival_order_ids,
        "arrival_order_times": arrival_order_times,
        "selected_particle_ids": selected_particle_ids,
        "selected_travel_times": selected_travel_times,
        "stopping_particle_id": stopping_particle_id,
        "stopping_travel_time": stopping_travel_time,
        "n_particles": n_particles,
        "n_arrived": int(arrival_order_ids.size),
        "arrival_fraction_reached": float(arrival_order_ids.size / n_particles) if n_particles else 0.0,
        "target_arrivals": target_count,
        "threshold_reached": bool(threshold_reached),
    }


def extract_particle_trajectories(px, py, pt, L, particle_ids):
    """Extract full trajectory records for the selected particle ids."""
    particle_ids = np.asarray(particle_ids, dtype=np.int32).ravel()
    trajectories = []
    for pid in particle_ids:
        k = int(L[pid])
        trajectories.append(
            {
                "particle_id": int(pid),
                "x": px[pid, :k].copy(),
                "y": py[pid, :k].copy(),
                "t": pt[pid, :k].copy(),
            }
        )
    return trajectories


def fastest_particle_id(travel_times):
    """Return the fastest successful particle id from a travel-time array."""
    travel_times = np.asarray(travel_times)
    if np.all(np.isnan(travel_times)):
        raise RuntimeError("No successful trajectories with valid travel time.")
    return int(np.nanargmin(travel_times))


def build_fastest_result_from_arrays(px, py, pt, L, travel_times):
    """Build the canonical fastest-result payload from dense trajectory arrays."""
    pid = fastest_particle_id(travel_times)
    k = int(L[pid])
    return {
        "particle_id": pid,
        "travel_time": float(travel_times[pid]),
        "x": px[pid, :k].copy(),
        "y": py[pid, :k].copy(),
        "t": pt[pid, :k].copy(),
    }


def build_fastest_result_from_trajectory(particle_id, travel_time, trajectory):
    """Build the canonical fastest-result payload from a single replayed trajectory dict."""
    return {
        "particle_id": int(particle_id),
        "travel_time": float(travel_time),
        "x": trajectory["x"],
        "y": trajectory["y"],
        "t": trajectory["t"],
    }


def fastest_trajectory(px, py, pt, L, st):
    nP = L.shape[0]

    # Valid if success and at least 1 point
    valid = (st == 0) & (L > 0)

    # Initialize with NaN so invalid ones don’t get picked
    travel_times = compute_travel_times(pt, L, st)

    result = build_fastest_result_from_arrays(px, py, pt, L, travel_times)
    result["all_travel_times"] = travel_times  # useful for histograms, optional
    return result

def fastest_trajectory_with_snapshot(px, py, pt, L, st):
    """
    Same as fastest_trajectory plus plume snapshot at t0.

    Finds the fastest trajectory and its first-passage time t0, then for every
    particle interpolates (x, y) at time t0. Returns the usual fastest_trajectory
    dict plus "t0", "x_at_t0", "y_at_t0" (NaN where particle was inactive at t0).
    """
    base = fastest_trajectory(px, py, pt, L, st)
    t0 = base["travel_time"]
    nP = L.shape[0]

    x_at_t0 = np.full(nP, np.nan, dtype=px.dtype)
    y_at_t0 = np.full(nP, np.nan, dtype=py.dtype)

    for i in range(nP):
        k = L[i]
        if k < 2:
            continue
        times = pt[i, :k]
        if times[0] <= t0 <= times[-1]:
            x_at_t0[i] = np.interp(t0, times, px[i, :k])
            y_at_t0[i] = np.interp(t0, times, py[i, :k])

    base["t0"] = t0
    base["x_at_t0"] = x_at_t0
    base["y_at_t0"] = y_at_t0
    return base
