"""
Compare fastest paths from PAR2, RK45 CPU, and RK45 GPU.

Workflow:
1) Generate fBm log-K field.
2) Solve velocity field with MODFLOW.
3) Run particle tracking: PAR2, RK45 CPU, RK45 GPU.
4) Compare fastest paths; visualize and save outputs.
"""

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
from simulation.hydraulic import read_mt3d_file, run_modflow
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


def _save_comparison_figure(
    path_fig,
    logk_field,
    vx,
    vy,
    rk45_cpu_res,
    rk45_gpu_res,
    par2_trajs,
    par2_fast,
):
    """Save a figure comparing PAR2, RK45 CPU, and RK45 GPU fastest paths."""
    log10k_field = logk_field / np.log(10.0)
    ny, nx = vx.shape
    x = np.arange(nx)
    y = np.arange(ny)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(log10k_field, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, label="log10 K", fraction=0.046, pad=0.04)

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

    ax.set_title("Fastest Path Comparison: PAR2 vs RK45 CPU vs RK45 GPU")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)


def main():
    # --- Paths ---
    path_cwd = Path(__file__).parent.resolve()
    path_out = path_cwd / "compare_output"
    path_mfoutput = path_cwd / "mf2005_output"
    path_p2input = path_cwd / "par2_input"
    path_p2output = path_cwd / "par2_output"

    for p in [path_out, path_mfoutput, path_p2input, path_p2output]:
        p.mkdir(parents=True, exist_ok=True)

    # --- Executables ---
    path_mf2005 = _resolve_executable("MODFLOW_EXE", "~/bin/mf2005", "MODFLOW")
    path_par2 = _resolve_executable("PAR2_EXE", "~/bin/par2", "PAR2")

    # --- Domain / fBm parameters ---
    np.random.seed(547)
    num_pt = 100
    dx = 1.0
    dy = 1.0
    dz = 1.0
    grad_head = 0.1
    hurst = 0.25
    std = 2.0
    minpadding = 1
    alpha = hurst * 2.0
    c = compute_c(num_pt, hurst, std)

    fbm_gen = fBmGenerator(c=c, alpha=alpha, num_pt=num_pt, dim=2, minpadding=minpadding)
    logk_field, msg_k = generate_logk_field(fbm_gen)
    print(msg_k)
    if logk_field is None or logk_field.ndim != 2:
        raise RuntimeError("Failed to generate a valid 2D log-K field.")

    ly, lx = logk_field.shape
    np.savetxt(path_out / "logk_field.dat", logk_field, delimiter=" ")

    # --- Source / target region setup (same style as simulation/main.py) ---
    source_vertex = [
        [lx // 20 - 1, ly // 20 - 1, 0],
        [lx // 20 - 1, ly - ly // 20 - 1, 1],
    ]
    target_vertex = [
        [lx - lx // 20 - 1, 0, 0],
        [lx - lx // 20 - 1, ly - 1, 1],
    ]
    x_cp = target_vertex[0][0]

    # --- MODFLOW ---
    ftl_path = path_mfoutput / "mf2005_model_modified.ftl"
    mf2005_config = {
        "modflow_exe": str(path_mf2005),
        "model_ws": str(path_mfoutput),
        "model_name": "mf2005_model",
        "outftl_name": str(ftl_path),
        "grad_head": grad_head,
        "lx": int(lx),
        "ly": int(ly),
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

    # --- PAR2 config and run ---
    update_config_file(
        str(path_p2input / "par2_config.yaml"),
        create_par2_config(dx, dy, lx, ly, path_mfoutput, path_p2output, source_vertex, x_cp),
    )
    return_code_p2, return_msg_p2 = run_par2(path_par2, path_p2input, path_p2output)
    print(return_msg_p2)
    par2_ok = return_code_p2 in (0, 10, 11)
    if not par2_ok:
        print("Warning: PAR2 did not succeed (e.g. no GPU). Continuing with RK45 only.")

    # --- Velocity field from MODFLOW output ---
    velocity_field = read_mt3d_file(str(ftl_path))
    if "QXX" not in velocity_field or "QYY" not in velocity_field:
        raise KeyError("FTL file does not contain expected QXX/QYY velocity fields.")
    vx = velocity_field["QXX"]
    vy = velocity_field["QYY"]

    # --- RK45 particle tracking ---
    x_grid = np.arange(vx.shape[1], dtype=np.float32) * np.float32(dx)
    y_grid = np.arange(vx.shape[0], dtype=np.float32) * np.float32(dy)

    x0 = float(source_vertex[0][0])
    y0 = float(source_vertex[0][1])
    y1 = float(source_vertex[1][1])
    density_particle = 20
    nP = max(2, int(density_particle * abs(y1 - y0)))
    seeds_x = np.full(nP, x0, dtype=np.float32)
    seeds_y = np.linspace(y0, y1, nP).astype(np.float32)

    args_particle_tracking = {
        "x": x_grid,
        "y": y_grid,
        "vx": vx,
        "vy": vy,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": float(x_cp),
        "eps_speed": 1e-30,
        "ds_min": 1e-20,
        "ds_max": 0.5 * min(dx, dy),
        "tol": 1e-4,
        "y_policy": "stop",
        "max_points": 4000,
        "max_iters": 200000,
    }
    px, py, pt, L, st = particle_tracking_cpu(args_particle_tracking)
    rk45_cpu_res = fastest_trajectory(px, py, pt, L, st)

    # --- RK45 GPU (optional; skip if no CUDA device) ---
    rk45_gpu_res = None
    try:
        px_g, py_g, pt_g, L_g, st_g, *_ = gpu_track_particles_arclength(
            x_grid,
            y_grid,
            vx,
            vy,
            seeds_x,
            seeds_y,
            right_x=float(x_cp),
            eps_speed=1e-30,
            ds_min=1e-20,
            ds_max=0.5 * min(dx, dy),
            tol=1e-4,
            y_policy="stop",
            max_points=4000,
            max_iters=200000,
        )
        rk45_gpu_res = fastest_trajectory(px_g, py_g, pt_g, L_g, st_g)
        print("RK45 GPU fastest path computed.")
    except Exception as e:
        print(f"Warning: RK45 GPU skipped ({e}). Continuing with CPU and PAR2 only.")

    # --- PAR2 trajectories (optional if PAR2 failed, e.g. no GPU) ---
    path_debut = path_p2output / "debut_trajectory"
    par2_trajs, par2_fast, par2_fast_time = _load_par2_debut_trajectories(path_debut, max_traj=5)
    if par2_fast is None:
        par2_fast_time = np.nan

    # --- Save compare data ---
    save_kw = {
        "logk_field": logk_field,
        "vx": vx,
        "vy": vy,
        "rk45_cpu_x": rk45_cpu_res["x"],
        "rk45_cpu_y": rk45_cpu_res["y"],
        "rk45_cpu_time": rk45_cpu_res["travel_time"],
        "par2_time": par2_fast_time,
    }
    if rk45_gpu_res is not None:
        save_kw["rk45_gpu_x"] = rk45_gpu_res["x"]
        save_kw["rk45_gpu_y"] = rk45_gpu_res["y"]
        save_kw["rk45_gpu_time"] = rk45_gpu_res["travel_time"]
    else:
        save_kw["rk45_gpu_x"] = np.array([], dtype=np.float64)
        save_kw["rk45_gpu_y"] = np.array([], dtype=np.float64)
        save_kw["rk45_gpu_time"] = np.nan
    if par2_fast is not None:
        save_kw["par2_x"] = par2_fast[:, 0]
        save_kw["par2_y"] = par2_fast[:, 1]
    else:
        save_kw["par2_x"] = np.array([], dtype=np.float64)
        save_kw["par2_y"] = np.array([], dtype=np.float64)
    np.savez(path_out / "compare_par2_rk45_data.npz", **save_kw)

    # --- Save figure ---
    fig_path = path_out / "compare_par2_rk45.png"
    _save_comparison_figure(
        fig_path,
        logk_field,
        vx,
        vy,
        rk45_cpu_res,
        rk45_gpu_res,
        par2_trajs,
        par2_fast,
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {path_out / 'compare_par2_rk45_data.npz'}")


if __name__ == "__main__":
    main()
