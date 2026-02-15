"""
Particle-tracking utilities: snapshot-at-t0 for plume visualization.
"""

import numpy as np

def fastest_trajectory(px, py, pt, L, st):
    nP = L.shape[0]

    # Valid if success and at least 1 point
    valid = (st == 0) & (L > 0)

    # Initialize with NaN so invalid ones don’t get picked
    travel_times = np.full(nP, np.nan, dtype=pt.dtype)

    # Extract final time for each valid trajectory
    idx = np.where(valid)[0]
    if idx.size > 0:
        # vectorized gather of last times
        travel_times[idx] = pt[idx, L[idx] - 1]

    # Find fastest (minimum travel time)
    if np.all(np.isnan(travel_times)):
        raise RuntimeError("No successful trajectories with valid travel time.")

    fastest_id = int(np.nanargmin(travel_times))
    fastest_time = float(travel_times[fastest_id])

    # Extract the polyline coordinates/time for the fastest trajectory
    k = L[fastest_id]
    x_fast = px[fastest_id, :k].copy()
    y_fast = py[fastest_id, :k].copy()
    t_fast = pt[fastest_id, :k].copy()

    return {
        "particle_id": fastest_id,
        "travel_time": fastest_time,
        "x": x_fast,
        "y": y_fast,
        "t": t_fast,
        "all_travel_times": travel_times  # useful for histograms, optional
    }

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
