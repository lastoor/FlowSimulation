"""
Compare the solved fastest path of a velocity field (from log-fBm conductivity) across solvers.

The velocity field is obtained by solving flow on a log-fBm conductivity field with MODFLOW.
We then compute the fastest path from source to target on that same velocity field using:
  - PAR2 (external executable)
  - rk45_cpu (Numba RK45 arclength)
  - rk45_gpu (Numba CUDA RK45 arclength)
  - rk4_fp (Taichi RK4 time stepping)

Workflow:
1) Generate log-fBm conductivity field.
2) Solve velocity field with MODFLOW from that conductivity.
3) Solve fastest path with PAR2, rk45_cpu, rk45_gpu, rk4_fp on the same velocity field.
4) Compare and visualize the four fastest paths; save figure and npz.
"""

import argparse
import os
import warnings
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simulation.config import create_par2_config, update_config_file
from simulation.grf_generation import compute_c, fBmGenerator, generate_logk_field
from simulation.hydraulic import read_mt3d_file, run_modflow, write_mt3d_file
from simulation.particle_tracking import (
    fastest_trajectory,
    particle_tracking_cpu,
    gpu_track_particles_arclength,
    run_par2,
)


def _resolve_executable(env_var, default_path, label):
    """Resolve executable path from env var or fallback."""
    env_path = os.environ.get(env_var, None)
    path = Path(env_path).expanduser().resolve() if env_path else Path(default_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"{label} executable not found at '{path}'. "
            f"Set {env_var} or install at {default_path}."
        )
    return path


def _load_par2_debut_trajectories(path_debut, max_traj=5):
    """
    Load debut trajectories from PAR2 output CSV files.

    Returns
    -------
    all_trajs : list[np.ndarray]
        Up to max_traj trajectories as arrays of shape (n, 2) for x,y.
    fastest_traj : np.ndarray | None
        Best trajectory by minimum final time if available, else first valid.
    fastest_time : float
        Final time of fastest_traj when available; np.nan otherwise.
    """
    traj_files = sorted(path_debut.glob("traj_*.csv"))
    if not traj_files:
        return [], None, np.nan

    all_trajs = []
    fastest_traj = None
    fastest_time = np.nan

    for file_ in traj_files[:max_traj]:
        raw = np.genfromtxt(file_, delimiter=",", skip_header=1)
        if raw.size == 0:
            continue
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 3:
            continue

        xy = raw[:, 1:3]
        all_trajs.append(xy)

        t_final = np.nan
        if raw.shape[1] >= 4 and np.all(np.isfinite(raw[:, -1])):
            t_final = float(raw[-1, -1])

        if fastest_traj is None:
            fastest_traj = xy
            fastest_time = t_final
        elif np.isfinite(t_final) and (not np.isfinite(fastest_time) or t_final < fastest_time):
            fastest_traj = xy
            fastest_time = t_final

    return all_trajs, fastest_traj, fastest_time


def _find_existing_file(path_dir, candidates):
    """Return first existing file in candidates under path_dir, else None."""
    for name in candidates:
        p = path_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def _first_crossing_path(x, y, t, x_cp):
    """
    Clip a trajectory to first x >= x_cp crossing in index coordinates.

    Returns (x_clip, y_clip, t_cross) or (None, None, np.nan) if no crossing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.size == 0 or y.size != x.size or t.size != x.size:
        return None, None, np.nan

    hit = np.where(x >= x_cp)[0]
    if hit.size == 0:
        return None, None, np.nan

    k = int(hit[0])
    if k == 0:
        return x[:1].copy(), y[:1].copy(), float(t[0])

    x0, y0, t0 = x[k - 1], y[k - 1], t[k - 1]
    x1, y1, t1 = x[k], y[k], t[k]
    if np.isfinite(x1 - x0) and abs(x1 - x0) > 1e-30:
        frac = (x_cp - x0) / (x1 - x0)
        frac = float(np.clip(frac, 0.0, 1.0))
    else:
        frac = 1.0

    y_cross = y0 + frac * (y1 - y0)
    t_cross = t0 + frac * (t1 - t0)
    x_clip = np.concatenate([x[:k], np.array([x_cp], dtype=np.float64)])
    y_clip = np.concatenate([y[:k], np.array([y_cross], dtype=np.float64)])
    return x_clip, y_clip, float(t_cross)


def _run_modpath7_and_extract_fastest(path_mp7, path_mfoutput, model_name, x0, y0, y1, nP, x_cp):
    """
    Run MODPATH7 with custom seed locations and extract fastest first-x_cp crossing.

    Returns a dict with keys x/y/travel_time or None on failure.
    """
    try:
        import flopy
    except Exception as e:
        print(f"Warning: MODPATH skipped (flopy import failed: {e}).")
        return None

    dis_file = path_mfoutput / f"{model_name}.dis"
    nam_file = path_mfoutput / f"{model_name}.nam"
    head_file = _find_existing_file(path_mfoutput, [f"{model_name}.hds", f"{model_name}.hed"])
    budget_file = _find_existing_file(path_mfoutput, [f"{model_name}.cbc", f"{model_name}.cbb", f"{model_name}.bud"])
    if head_file is None or budget_file is None or not dis_file.exists() or not nam_file.exists():
        missing = []
        if not nam_file.exists():
            missing.append(str(nam_file))
        if not dis_file.exists():
            missing.append(str(dis_file))
        if head_file is None:
            missing.append(f"{model_name}.hds/.hed")
        if budget_file is None:
            missing.append(f"{model_name}.cbc/.cbb/.bud")
        print(f"Warning: MODPATH skipped (missing MODFLOW artifacts: {', '.join(missing)}).")
        return None

    mp_name = f"{model_name}_mp7"
    try:
        mf = flopy.modflow.Modflow.load(
            nam_file.name,
            model_ws=str(path_mfoutput),
            check=False,
            verbose=False,
            forgive=False,
        )
        if mf is None:
            raise RuntimeError("Failed to load MODFLOW model for MODPATH setup.")

        mp = flopy.modpath.Modpath7(
            modelname=mp_name,
            exe_name=str(path_mp7),
            flowmodel=mf,
            headfilename=str(head_file),
            budgetfilename=str(budget_file),
            model_ws=str(path_mfoutput),
        )

        flopy.modpath.Modpath7Bas(mp, porosity=0.3)

        # Build seeds matching compare.py coordinates (cell-center index space).
        sx = np.full(int(nP), float(x0), dtype=np.float64)
        sy = np.linspace(float(y0), float(y1), int(nP), dtype=np.float64)

        nrow, ncol = mf.nrow, mf.ncol
        # MP7/MODFLOW rows are top-origin; comparison/plotting here are bottom-origin.
        x_phys = sx + 0.5
        y_phys_top = nrow - (sy + 0.5)
        j0 = np.clip(np.floor(x_phys).astype(np.int64), 0, ncol - 1)
        i0 = np.clip(np.floor(y_phys_top).astype(np.int64), 0, nrow - 1)
        xloc0 = np.clip(x_phys - j0, 1e-6, 1.0 - 1e-6).astype(np.float32)
        yloc0 = np.clip(y_phys_top - i0, 1e-6, 1.0 - 1e-6).astype(np.float32)

        partlocs = [(0, int(ii), int(jj)) for ii, jj in zip(i0.tolist(), j0.tolist())]
        pdat = flopy.modpath.ParticleData(
            partlocs=partlocs,
            structured=True,
            localx=xloc0,
            localy=yloc0,
            localz=np.full(int(nP), 0.5, dtype=np.float32),
        )
        pgroup = flopy.modpath.ParticleGroup(
            particlegroupname="PG1",
            particledata=pdat,
        )
        flopy.modpath.Modpath7Sim(
            mp,
            simulationtype="pathline",
            trackingdirection="forward",
            weaksinkoption="stop_at",
            weaksourceoption="stop_at",
            referencetime=0.0,
            budgetoutputoption="no",
            particlegroups=[pgroup],
        )

        mp.write_name_file()
        mp.write_input()
        # Some MP7 binaries require NAME file entries as: KEYWORD UNIT FILE.
        mpnam_path = path_mfoutput / mp.mpnamefile
        with open(mpnam_path, "w", encoding="ascii") as f:
            f.write("# Name file for MODPATH 7 (unit-based entries)\n")
            f.write(f"MPBAS      86 {mp_name}.mpbas\n")
            f.write(f"DIS        87 {dis_file.name}\n")
            f.write(f"HEAD       88 {head_file.name}\n")
            f.write(f"BUDGET     89 {budget_file.name}\n")
        ok, buff = mp.run_model(silent=True, report=True)
        if not ok:
            msg = "; ".join(buff[-5:]) if isinstance(buff, list) and buff else "unknown MODPATH error"
            print(f"Warning: MODPATH7 skipped (run failed: {msg}).")
            return None

        pth_path = path_mfoutput / f"{mp_name}.mppth"
        if not pth_path.exists():
            print("Warning: MODPATH7 skipped (pathline file not found).")
            return None
        pthobj = flopy.utils.PathlineFile(str(pth_path))
        rec = pthobj.get_alldata()
    except Exception as e:
        print(f"Warning: MODPATH7 skipped ({e}).")
        return None

    if isinstance(rec, list):
        rec = np.concatenate([r for r in rec if r is not None and len(r) > 0]) if rec else None
    if rec is None or len(rec) == 0:
        print("Warning: MODPATH7 skipped (no pathline records).")
        return None

    fastest = None
    fastest_time = np.inf
    for pid in np.unique(rec["particleid"]):
        rows = rec[rec["particleid"] == pid]
        if rows.size < 2:
            continue
        order = np.argsort(rows["time"])
        x = np.asarray(rows["x"][order], dtype=np.float64) - 0.5
        y = nrow - np.asarray(rows["y"][order], dtype=np.float64) - 0.5
        t = np.asarray(rows["time"][order], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x, y, t = x[valid], y[valid], t[valid]
        if x.size < 2:
            continue

        x_clip, y_clip, t_cross = _first_crossing_path(x, y, t, x_cp=float(x_cp))
        if x_clip is None or not np.isfinite(t_cross):
            continue
        if t_cross < fastest_time:
            fastest_time = float(t_cross)
            fastest = {
                "particle_id": int(pid),
                "travel_time": float(t_cross),
                "x": x_clip,
                "y": y_clip,
            }

    if fastest is None:
        print("Warning: MODPATH7 finished but no trajectory crossed x_cp.")
    else:
        print("MODPATH7 fastest path computed.")
    return fastest


def _save_comparison_figure(
    path_fig,
    background_field,
    background_label,
    title,
    vx,
    vy,
    rk45_cpu_res,
    rk45_gpu_res,
    rk4_fp_res,
    modpath_res,
    par2_trajs,
    par2_fast,
    expected_lane=None,
):
    """Save a figure comparing PAR2, RK45 CPU/GPU, and RK4 FP fastest paths."""
    ny, nx = vx.shape
    x = np.arange(nx)
    y = np.arange(ny)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(background_field, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, label=background_label, fraction=0.046, pad=0.04)

    ax.streamplot(
        x,
        y,
        vx,
        vy,
        color=to_rgba("black", alpha=0.35),
        linewidth=0.7,
        arrowsize=0.6,
        density=1.1,
    )

    for idx, traj in enumerate(par2_trajs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="#444444",
            lw=1.0,
            alpha=0.8,
            label="PAR2 debut trajectories" if idx == 0 else None,
        )

    if par2_fast is not None:
        ax.plot(
            par2_fast[:, 0],
            par2_fast[:, 1],
            color="#D62728",
            lw=2.0,
            label="Fastest Path (PAR2)",
        )

    if expected_lane is not None:
        ax.plot(
            expected_lane[:, 0],
            expected_lane[:, 1],
            color="#FFD700",
            lw=2.0,
            ls=":",
            label="Expected fastest lane",
        )

    if rk45_cpu_res is not None:
        ax.plot(
            rk45_cpu_res["x"],
            rk45_cpu_res["y"],
            "w--",
            lw=2.0,
            label=f'Fastest Path (RK45 CPU), t={rk45_cpu_res["travel_time"]:.3g}',
        )

    if rk45_gpu_res is not None:
        ax.plot(
            rk45_gpu_res["x"],
            rk45_gpu_res["y"],
            "c-.",
            lw=2.0,
            label=f'Fastest Path (RK45 GPU), t={rk45_gpu_res["travel_time"]:.3g}',
        )

    if rk4_fp_res is not None:
        ax.plot(
            rk4_fp_res["x"],
            rk4_fp_res["y"],
            "m-",
            lw=2.0,
            label=f'Fastest Path (RK4 FP Taichi), t={rk4_fp_res["travel_time"]:.3g}',
        )

    if modpath_res is not None:
        ax.plot(
            modpath_res["x"],
            modpath_res["y"],
            color="#FFA500",
            lw=2.0,
            label=f'Fastest Path (MODPATH7), t={modpath_res["travel_time"]:.3g}',
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare fastest path across solvers on either fbm->MODFLOW or synthetic velocity fields.",
    )
    p.add_argument(
        "--test-case",
        choices=("fbm", "synthetic", "both"),
        default="fbm",
        help="Velocity field source: fbm (default), synthetic, or both.",
    )
    p.add_argument(
        "--synthetic-shape-tol",
        type=float,
        default=2.0,
        help="Max mean nearest-distance (grid cells) to expected fastest lane in synthetic mode.",
    )
    p.add_argument(
        "--synthetic-time-gap-min",
        type=float,
        default=5e-3,
        help="Minimum expected time gap between fastest and second-fastest synthetic lanes.",
    )
    p.add_argument(
        "--no-par2",
        action="store_true",
        help="Do not run PAR2 in the comparison.",
    )
    p.add_argument(
        "--no-rk45-cpu",
        action="store_true",
        help="Do not run RK45 CPU in the comparison.",
    )
    p.add_argument(
        "--no-rk45-gpu",
        action="store_true",
        help="Do not run RK45 GPU in the comparison.",
    )
    p.add_argument(
        "--no-rk4-fp",
        action="store_true",
        help="Do not run RK4 FP (Taichi) in the comparison.",
    )
    p.add_argument(
        "--no-modpath",
        action="store_true",
        help="Do not run MODPATH7 in the comparison.",
    )
    args = p.parse_args()
    use_par2 = not args.no_par2
    use_rk45_cpu = not args.no_rk45_cpu
    use_rk45_gpu = not args.no_rk45_gpu
    use_rk4_fp = not args.no_rk4_fp
    use_modpath = not args.no_modpath
    if not (use_par2 or use_rk45_cpu or use_rk45_gpu or use_rk4_fp or use_modpath):
        p.error("At least one algorithm must be enabled (do not pass all --no-* flags).")
    return argparse.Namespace(
        test_case=args.test_case,
        synthetic_shape_tol=float(args.synthetic_shape_tol),
        synthetic_time_gap_min=float(args.synthetic_time_gap_min),
        use_par2=use_par2,
        use_rk45_cpu=use_rk45_cpu,
        use_rk45_gpu=use_rk45_gpu,
        use_rk4_fp=use_rk4_fp,
        use_modpath=use_modpath,
    )


def _polyline_travel_time(x_path, y_path, vx, vy):
    """Estimate travel time along a path using local speed magnitude."""
    x_path = np.asarray(x_path, dtype=np.float64)
    y_path = np.asarray(y_path, dtype=np.float64)
    if x_path.size < 2 or y_path.size != x_path.size:
        return np.inf

    ny, nx = vx.shape
    xi = np.clip(np.rint(x_path).astype(np.int64), 0, nx - 1)
    yi = np.clip(np.rint(y_path).astype(np.int64), 0, ny - 1)
    speed = np.sqrt(vx[yi, xi] ** 2 + vy[yi, xi] ** 2)
    seg_speed = 0.5 * (speed[:-1] + speed[1:])
    dx = np.diff(x_path)
    dy = np.diff(y_path)
    ds = np.sqrt(dx * dx + dy * dy)
    return float(np.sum(ds / np.maximum(seg_speed, 1e-12)))


def _build_synthetic_velocity_field(nx, ny):
    """
    Build a deterministic velocity field with one unique fastest sine-like lane.
    """
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    Y = np.broadcast_to(y[:, None], (ny, nx))

    base_vx = 0.04
    vx = np.full((ny, nx), base_vx, dtype=np.float64)
    vy = np.zeros((ny, nx), dtype=np.float64)

    lanes = [
        {"name": "lane0", "y0": 0.28 * ny, "amp": 5.5, "freq": 1.2, "phase": 0.2, "boost": 0.05, "width": 2.4},
        {"name": "lane1", "y0": 0.52 * ny, "amp": 8.0, "freq": 2, "phase": 1.0, "boost": 50, "width": 2.2},
        {"name": "lane2", "y0": 0.72 * ny, "amp": 4.5, "freq": 0.9, "phase": -0.6, "boost": 0.05, "width": 2.6},
    ]

    lane_paths = []
    lane_times = []
    for lane in lanes:
        phase = (2.0 * np.pi * lane["freq"] * x / nx) + lane["phase"]
        y_curve = lane["y0"] + lane["amp"] * np.sin(phase)
        dydx = lane["amp"] * (2.0 * np.pi * lane["freq"] / nx) * np.cos(phase)
        dist = Y - y_curve[None, :]
        width = max(float(lane["width"]), 1e-6)
        weight = np.exp(-0.5 * (dist / width) ** 2)
        tangent_norm = np.sqrt(1.0 + dydx[None, :] ** 2)
        vx += weight * (lane["boost"] / tangent_norm)
        vy += weight * (lane["boost"] * dydx[None, :] / tangent_norm)

        y_curve = np.clip(y_curve, 0.0, ny - 1.0)
        path = np.column_stack((x, y_curve))
        lane_paths.append(path)
        lane_times.append(_polyline_travel_time(path[:, 0], path[:, 1], vx, vy))

    vx = np.maximum(vx, 1e-8)
    lane_times = np.asarray(lane_times, dtype=np.float64)
    order = np.argsort(lane_times)
    fastest_idx = int(order[0])
    second_idx = int(order[1])
    expected = {
        "lane_paths": lane_paths,
        "lane_times": lane_times,
        "fastest_lane_id": fastest_idx,
        "fastest_lane": lane_paths[fastest_idx],
        "second_fastest_time": float(lane_times[second_idx]),
    }
    return vx, vy, expected


def _downsample_path(path, max_points=2000):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    if path.shape[0] <= max_points:
        return path
    idx = np.linspace(0, path.shape[0] - 1, max_points).astype(np.int64)
    return path[idx]


def _mean_nearest_distance(path_a, path_b):
    """Mean nearest-neighbor distance from path_a to path_b."""
    pa = _downsample_path(path_a, max_points=2000)
    pb = _downsample_path(path_b, max_points=1500)
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return np.inf
    delta = pa[:, None, :] - pb[None, :, :]
    d2 = np.sum(delta * delta, axis=2)
    return float(np.mean(np.sqrt(np.min(d2, axis=1))))


def _evaluate_synthetic_solver_results(expected, solver_results, shape_tol, time_gap_min):
    """Evaluate synthetic-case success by shape proximity and travel time."""
    fastest_lane = expected["fastest_lane"]
    fastest_time = float(expected["lane_times"][expected["fastest_lane_id"]])
    second_fastest_time = float(expected["second_fastest_time"])
    if (second_fastest_time - fastest_time) <= float(time_gap_min):
        raise RuntimeError(
            "Synthetic profile is not uniquely fastest under configured time-gap criterion. "
            f"gap={second_fastest_time - fastest_time:.6g}, required>{time_gap_min:.6g}"
        )

    time_upper = fastest_time + max(0.25 * fastest_time, 0.5 * float(time_gap_min))
    print("\n--- Synthetic validation (shape + time) ---")
    print(
        f"  Expected fastest lane: id={expected['fastest_lane_id']}, "
        f"t*={fastest_time:.6g}, second={second_fastest_time:.6g}, gap={second_fastest_time-fastest_time:.6g}"
    )

    metrics = {}
    for solver_name, res in solver_results.items():
        key = solver_name.lower().replace(" ", "_")
        if res is None:
            metrics[f"{key}_shape_error"] = np.nan
            metrics[f"{key}_time_error"] = np.nan
            metrics[f"{key}_pass_flag"] = 0
            print(f"  {solver_name}: (not available)")
            continue

        path = np.column_stack((np.asarray(res["x"], dtype=np.float64), np.asarray(res["y"], dtype=np.float64)))
        shape_error = _mean_nearest_distance(path, fastest_lane)
        travel_time = float(res["travel_time"])
        time_error = travel_time - fastest_time

        pass_shape = shape_error <= float(shape_tol)
        pass_time = (travel_time <= (second_fastest_time - float(time_gap_min))) and (travel_time <= time_upper)
        pass_flag = int(pass_shape and pass_time)

        metrics[f"{key}_shape_error"] = float(shape_error)
        metrics[f"{key}_time_error"] = float(time_error)
        metrics[f"{key}_pass_flag"] = pass_flag
        print(
            f"  {solver_name}: "
            f"shape_err={shape_error:.4g} (<= {shape_tol:.4g}: {pass_shape}), "
            f"time={travel_time:.6g}, dt={time_error:.6g}, pass={bool(pass_flag)}"
        )
    print("---")
    return metrics


def _run_case(
    case_name,
    opts,
    path_out,
    path_mfoutput,
    path_p2input,
    path_p2output,
    path_par2,
    path_mp7,
    source_vertex,
    x_cp,
    vx,
    vy,
    dx,
    dy,
    x0,
    y0,
    y1,
    nP,
    dt_compare,
    max_steps_compare,
    right_x,
    eps_speed,
    ds_min,
    ds_max,
    tol,
    y_policy,
    max_points,
    max_iters,
    max_traj,
    background_field,
    background_label,
    title,
    expected_synthetic=None,
    modpath_model_name=None,
):
    ly, lx = vx.shape
    ftl_path = path_mfoutput / ("mf2005_model_modified.ftl" if case_name == "fbm" else "synthetic_velocity.ftl")

    if case_name == "synthetic":
        cnh = []
        for j in range(ly):
            cnh.append([1, j + 1, 1, 1.0e-2])
            cnh.append([1, j + 1, lx, 0.0])
        vars_ftl = {
            "header": " 'MT3D4.00.00'           0           0           0           0           0           0         200           1           1           0           0           0           0           0           0           0           0           0           0           0           0",
            "THKSAT": -111.0 * np.ones((ly, lx), dtype=np.float64),
            "QXX": np.asarray(vx, dtype=np.float64),
            "QYY": np.asarray(vy, dtype=np.float64),
            "CNH": np.asarray(cnh, dtype=np.float64),
        }
        par2_overrides = {
            "physics": {
                "velocity": {
                    "file": str(ftl_path),
                },
            },
        }
        write_mt3d_file(str(ftl_path), vars_ftl)

    par2_ok = False
    if opts.use_par2:
        par2_overrides = par2_overrides if "par2_overrides" in locals() else {}
        par2_overrides.update(
            {
                "simulation": {
                    "particles": {"N": nP},
                    "dt": dt_compare,
                    "steps": max_steps_compare,
                },
                "output": {"debut": {"num": max_traj}},
            }
        )
        update_config_file(
            str(path_p2input / "par2_config.yaml"),
            create_par2_config(
                dx, dy, lx, ly, path_mfoutput, path_p2output, source_vertex, x_cp,
                overrides=par2_overrides,
            ),
        )
        return_code_p2, return_msg_p2 = run_par2(path_par2, path_p2input, path_p2output)
        print(return_msg_p2)
        par2_ok = return_code_p2 in (0, 10, 11)
        if not par2_ok:
            print("Warning: PAR2 did not succeed (e.g. no GPU). Continuing with others.")

    x_grid = np.arange(vx.shape[1], dtype=np.float32) * np.float32(dx)
    y_grid = np.arange(vx.shape[0], dtype=np.float32) * np.float32(dy)
    seeds_x = np.full(nP, x0, dtype=np.float32)
    seeds_y = np.linspace(y0, y1, nP).astype(np.float32)

    args_particle_tracking = {
        "x": x_grid,
        "y": y_grid,
        "vx": np.asarray(vx, dtype=np.float64),
        "vy": np.asarray(vy, dtype=np.float64),
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": right_x,
        "velocity_location": "face",
        "eps_speed": eps_speed,
        "ds_min": ds_min,
        "ds_max": ds_max,
        "tol": tol,
        "y_policy": y_policy,
        "max_points": max_points,
        "max_iters": max_iters,
        "dt": dt_compare,
        "max_steps": max_steps_compare,
    }
    rk45_cpu_res = None
    if opts.use_rk45_cpu:
        px, py, pt, L, st = particle_tracking_cpu(args_particle_tracking)
        rk45_cpu_res = fastest_trajectory(px, py, pt, L, st)

    rk45_gpu_res = None
    if opts.use_rk45_gpu:
        if gpu_track_particles_arclength is None:
            print("Warning: RK45 GPU skipped (GPU backend unavailable in this environment).")
        else:
            try:
                p = args_particle_tracking
                px_g, py_g, pt_g, L_g, st_g, *_ = gpu_track_particles_arclength(
                    p["x"],
                    p["y"],
                    p["vx"],
                    p["vy"],
                    p["seeds_x"],
                    p["seeds_y"],
                    right_x=p["right_x"],
                    eps_speed=p["eps_speed"],
                    ds_min=p["ds_min"],
                    ds_max=p["ds_max"],
                    tol=p["tol"],
                    y_policy=p["y_policy"],
                    max_points=p["max_points"],
                    max_iters=p["max_iters"],
                )
                rk45_gpu_res = fastest_trajectory(px_g, py_g, pt_g, L_g, st_g)
                print("RK45 GPU fastest path computed.")
            except Exception as e:
                print(f"Warning: RK45 GPU skipped ({e}).")

    rk4_fp_res = None
    if opts.use_rk4_fp:
        try:
            from simulation.particle_tracking.rk4_fp import rk4_fastest_path_params

            rk4_fp_res = rk4_fastest_path_params(args_particle_tracking)
            print("RK4 FP (Taichi) fastest path computed.")
        except Exception as e:
            print(f"Warning: RK4 FP skipped ({e}).")

    modpath_res = None
    if opts.use_modpath and modpath_model_name is not None:
        modpath_res = _run_modpath7_and_extract_fastest(
            path_mp7=path_mp7,
            path_mfoutput=path_mfoutput,
            model_name=modpath_model_name,
            x0=x0,
            y0=y0,
            y1=y1,
            nP=nP,
            x_cp=x_cp,
        )

    par2_trajs = []
    par2_fast = None
    par2_fast_time = np.nan
    if opts.use_par2:
        path_debut = path_p2output / "debut_trajectory"
        par2_trajs, par2_fast, par2_fast_time = _load_par2_debut_trajectories(path_debut, max_traj=max_traj)
        if par2_fast is None:
            par2_fast_time = np.nan

    print(f"\n--- Fastest path travel times ({case_name}) ---")
    if opts.use_par2:
        print(f"  PAR2:      {par2_fast_time:.6g}" if np.isfinite(par2_fast_time) else "  PAR2:      (not available)")
    if opts.use_rk45_cpu and rk45_cpu_res is not None:
        print(f"  RK45 CPU:  {rk45_cpu_res['travel_time']:.6g}")
    if opts.use_rk45_gpu:
        print(f"  RK45 GPU:  {rk45_gpu_res['travel_time']:.6g}" if rk45_gpu_res is not None else "  RK45 GPU:  (skipped)")
    if opts.use_rk4_fp:
        print(f"  RK4 FP:    {rk4_fp_res['travel_time']:.6g}" if rk4_fp_res is not None else "  RK4 FP:    (skipped)")
    if opts.use_modpath and modpath_model_name is not None:
        print(f"  MODPATH7:  {modpath_res['travel_time']:.6g}" if modpath_res is not None else "  MODPATH7:  (skipped)")
    print("---")

    save_kw = {
        "case_name": np.array([case_name]),
        "vx": np.asarray(vx, dtype=np.float64),
        "vy": np.asarray(vy, dtype=np.float64),
        "par2_time": par2_fast_time,
    }
    if rk45_cpu_res is not None:
        save_kw["rk45_cpu_x"] = rk45_cpu_res["x"]
        save_kw["rk45_cpu_y"] = rk45_cpu_res["y"]
        save_kw["rk45_cpu_time"] = rk45_cpu_res["travel_time"]
    else:
        save_kw["rk45_cpu_x"] = np.array([], dtype=np.float64)
        save_kw["rk45_cpu_y"] = np.array([], dtype=np.float64)
        save_kw["rk45_cpu_time"] = np.nan
    if rk45_gpu_res is not None:
        save_kw["rk45_gpu_x"] = rk45_gpu_res["x"]
        save_kw["rk45_gpu_y"] = rk45_gpu_res["y"]
        save_kw["rk45_gpu_time"] = rk45_gpu_res["travel_time"]
    else:
        save_kw["rk45_gpu_x"] = np.array([], dtype=np.float64)
        save_kw["rk45_gpu_y"] = np.array([], dtype=np.float64)
        save_kw["rk45_gpu_time"] = np.nan
    if rk4_fp_res is not None:
        save_kw["rk4_fp_x"] = rk4_fp_res["x"]
        save_kw["rk4_fp_y"] = rk4_fp_res["y"]
        save_kw["rk4_fp_time"] = rk4_fp_res["travel_time"]
    else:
        save_kw["rk4_fp_x"] = np.array([], dtype=np.float64)
        save_kw["rk4_fp_y"] = np.array([], dtype=np.float64)
        save_kw["rk4_fp_time"] = np.nan
    if par2_fast is not None:
        save_kw["par2_x"] = par2_fast[:, 0]
        save_kw["par2_y"] = par2_fast[:, 1]
    else:
        save_kw["par2_x"] = np.array([], dtype=np.float64)
        save_kw["par2_y"] = np.array([], dtype=np.float64)
    if modpath_res is not None:
        save_kw["modpath_x"] = np.asarray(modpath_res["x"], dtype=np.float64)
        save_kw["modpath_y"] = np.asarray(modpath_res["y"], dtype=np.float64)
        save_kw["modpath_time"] = float(modpath_res["travel_time"])
    else:
        save_kw["modpath_x"] = np.array([], dtype=np.float64)
        save_kw["modpath_y"] = np.array([], dtype=np.float64)
        save_kw["modpath_time"] = np.nan

    if expected_synthetic is not None:
        expected_lane = expected_synthetic["fastest_lane"]
        save_kw["synthetic_expected_lane_id"] = expected_synthetic["fastest_lane_id"]
        save_kw["synthetic_expected_lane_x"] = expected_lane[:, 0]
        save_kw["synthetic_expected_lane_y"] = expected_lane[:, 1]
        save_kw["synthetic_lane_times"] = expected_synthetic["lane_times"]
        save_kw["synthetic_shape_tol"] = float(opts.synthetic_shape_tol)
        save_kw["synthetic_time_gap_min"] = float(opts.synthetic_time_gap_min)
        metrics = _evaluate_synthetic_solver_results(
            expected_synthetic,
            {
                "PAR2": None if par2_fast is None else {"x": par2_fast[:, 0], "y": par2_fast[:, 1], "travel_time": par2_fast_time},
                "RK45 CPU": rk45_cpu_res,
                "RK45 GPU": rk45_gpu_res,
                "RK4 FP": rk4_fp_res,
            },
            shape_tol=opts.synthetic_shape_tol,
            time_gap_min=opts.synthetic_time_gap_min,
        )
        save_kw.update(metrics)
    else:
        expected_lane = None

    npz_path = path_out / f"compare_par2_rk45_data_{case_name}.npz"
    np.savez(npz_path, **save_kw)
    if case_name == "fbm":
        np.savez(path_out / "compare_par2_rk45_data.npz", **save_kw)

    fig_path = path_out / f"compare_par2_rk45_rk4fp_{case_name}.png"
    _save_comparison_figure(
        fig_path,
        background_field=background_field,
        background_label=background_label,
        title=title,
        vx=vx,
        vy=vy,
        rk45_cpu_res=rk45_cpu_res,
        rk45_gpu_res=rk45_gpu_res,
        rk4_fp_res=rk4_fp_res,
        modpath_res=modpath_res,
        par2_trajs=par2_trajs,
        par2_fast=par2_fast,
        expected_lane=expected_lane,
    )
    if case_name == "fbm":
        _save_comparison_figure(
            path_out / "compare_par2_rk45_rk4fp.png",
            background_field=background_field,
            background_label=background_label,
            title=title,
            vx=vx,
            vy=vy,
            rk45_cpu_res=rk45_cpu_res,
            rk45_gpu_res=rk45_gpu_res,
            rk4_fp_res=rk4_fp_res,
            modpath_res=modpath_res,
            par2_trajs=par2_trajs,
            par2_fast=par2_fast,
            expected_lane=expected_lane,
        )
    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {npz_path}")


def main():
    opts = _parse_args()

    path_cwd = Path(__file__).parent.resolve()
    path_out = path_cwd / "compare_output"
    path_mfoutput = path_cwd / "mf2005_output"
    path_p2input = path_cwd / "par2_input"
    path_p2output = path_cwd / "par2_output"
    for p in [path_out, path_mfoutput, path_p2input, path_p2output]:
        p.mkdir(parents=True, exist_ok=True)

    np.random.seed(6238771)
    num_pt = 100
    dx = 1.0
    dy = 1.0
    dz = 1.0
    grad_head = 0.1

    run_fbm = opts.test_case in ("fbm", "both")
    run_synthetic = opts.test_case in ("synthetic", "both")

    path_mf2005 = _resolve_executable("MODFLOW_EXE", "~/bin/mf2005", "MODFLOW") if run_fbm else None
    path_par2 = _resolve_executable("PAR2_EXE", "~/bin/par2", "PAR2") if opts.use_par2 else None
    path_mp7 = _resolve_executable("MODPATH_EXE", "~/bin/mp7", "MODPATH7") if (opts.use_modpath and run_fbm) else None

    ly = num_pt
    lx = num_pt
    source_vertex = [
        [lx // 20 - 1, ly // 20 - 1, 0],
        [lx // 20 - 1, ly - ly // 20 - 1, 1],
    ]
    target_vertex = [
        [lx - lx // 20 - 1, 0, 0],
        [lx - lx // 20 - 1, ly - 1, 1],
    ]
    x_cp = target_vertex[0][0]
    x0 = float(source_vertex[0][0])
    y0 = float(source_vertex[0][1])
    y1 = float(source_vertex[1][1])
    density_particle = 10
    nP = max(2, int(density_particle * abs(y1 - y0)))
    dt_compare = 0.001 * min(dx, dy)
    max_steps_compare = 200000
    right_x = float(x_cp)
    eps_speed = 1e-30
    ds_min = 1e-20
    ds_max = 0.5 * min(dx, dy)
    tol = 1e-4
    y_policy = "stop"
    max_points = 100000
    max_iters = 200000
    max_traj = 5

    if run_fbm:
        hurst = 0.25
        std = 3.0
        minpadding = 1
        alpha = hurst * 2.0
        c = compute_c(num_pt, hurst, std)
        fbm_gen = fBmGenerator(c=c, alpha=alpha, num_pt=num_pt, dim=2, minpadding=minpadding)
        logk_field, msg_k = generate_logk_field(fbm_gen)
        print(msg_k)
        if logk_field is None or logk_field.ndim != 2:
            raise RuntimeError("Failed to generate a valid 2D log-K field.")
        np.savetxt(path_out / "logk_field.dat", logk_field, delimiter=" ")

        mf2005_config = {
            "modflow_exe": str(path_mf2005),
            "model_ws": str(path_mfoutput),
            "model_name": "mf2005_model",
            "outftl_name": str(path_mfoutput / "mf2005_model_modified.ftl"),
            "grad_head": grad_head,
            "lx": int(logk_field.shape[1]),
            "ly": int(logk_field.shape[0]),
            "lz": 1,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "logk_field": logk_field,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return_msg_mf = run_modflow(mf2005_config)
        print(return_msg_mf)
        if "successfully" not in return_msg_mf.lower():
            raise RuntimeError(f"MODFLOW run failed: {return_msg_mf}")

        velocity_field = read_mt3d_file(str(path_mfoutput / "mf2005_model_modified.ftl"))
        if "QXX" not in velocity_field or "QYY" not in velocity_field:
            raise KeyError("FTL file does not contain expected QXX/QYY velocity fields.")
        vx_fbm = velocity_field["QXX"]
        vy_fbm = velocity_field["QYY"]

        _run_case(
            case_name="fbm",
            opts=opts,
            path_out=path_out,
            path_mfoutput=path_mfoutput,
            path_p2input=path_p2input,
            path_p2output=path_p2output,
            path_par2=path_par2,
            path_mp7=path_mp7,
            source_vertex=source_vertex,
            x_cp=x_cp,
            vx=vx_fbm,
            vy=vy_fbm,
            dx=dx,
            dy=dy,
            x0=x0,
            y0=y0,
            y1=y1,
            nP=nP,
            dt_compare=dt_compare,
            max_steps_compare=max_steps_compare,
            right_x=right_x,
            eps_speed=eps_speed,
            ds_min=ds_min,
            ds_max=ds_max,
            tol=tol,
            y_policy=y_policy,
            max_points=max_points,
            max_iters=max_iters,
            max_traj=max_traj,
            background_field=logk_field / np.log(10.0),
            background_label="log10 K",
            title="Fastest path (velocity from log-fBm conductivity): PAR2 vs RK45 CPU/GPU vs RK4 FP vs MODPATH7",
            expected_synthetic=None,
            modpath_model_name="mf2005_model",
        )

    if run_synthetic:
        vx_syn, vy_syn, expected = _build_synthetic_velocity_field(nx=lx, ny=ly)
        if (expected["second_fastest_time"] - expected["lane_times"][expected["fastest_lane_id"]]) <= opts.synthetic_time_gap_min:
            raise RuntimeError(
                "Synthetic field does not satisfy unique fastest lane under current --synthetic-time-gap-min."
            )
    
        ftl_path = path_mfoutput / "synthetic_velocity.ftl"
        cnh = []
        for j in range(ly):
            cnh.append([1, j + 1, 1, 1.0e-2])
            cnh.append([1, j + 1, lx, 0.0])
        vars_ftl = {
            "header": " 'MT3D4.00.00'           0           0           0           0           0           0         200           1           1           0           0           0           0           0           0           0           0           0           0           0           0",
            "THKSAT": -111.0 * np.ones((ly, lx), dtype=np.float64),
            "QXX": np.asarray(vx_syn, dtype=np.float64),
            "QYY": np.asarray(vy_syn, dtype=np.float64),
            "CNH": np.asarray(cnh, dtype=np.float64),
        }
        write_mt3d_file(str(ftl_path), vars_ftl)

        _run_case(
            case_name="synthetic",
            opts=opts,
            path_out=path_out,
            path_mfoutput=path_mfoutput,
            path_p2input=path_p2input,
            path_p2output=path_p2output,
            path_par2=path_par2,
            path_mp7=path_mp7,
            source_vertex=source_vertex,
            x_cp=x_cp,
            vx=vx_syn,
            vy=vy_syn,
            dx=dx,
            dy=dy,
            x0=x0,
            y0=y0,
            y1=y1,
            nP=nP,
            dt_compare=dt_compare,
            max_steps_compare=max_steps_compare,
            right_x=right_x,
            eps_speed=eps_speed,
            ds_min=ds_min,
            ds_max=ds_max,
            tol=tol,
            y_policy=y_policy,
            max_points=max_points,
            max_iters=max_iters,
            max_traj=max_traj,
            background_field=np.sqrt(vx_syn * vx_syn + vy_syn * vy_syn),
            background_label="|v|",
            title="Fastest path on synthetic velocity field: PAR2 vs RK45 CPU/GPU vs RK4 FP",
            expected_synthetic=expected,
            modpath_model_name=None,
        )


if __name__ == "__main__":
    main()
