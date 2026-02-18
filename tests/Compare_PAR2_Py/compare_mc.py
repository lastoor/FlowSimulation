"""
Monte Carlo comparison of PAR2 and RK45 fastest paths.

Each realization follows the same workflow as compare.py:
1) Generate fBm log-K field.
2) Solve flow with MODFLOW.
3) Track particles with PAR2 and RK45 (CPU).
4) Measure mismatch between fastest paths using:
   - symmetric Hausdorff distance,
   - 2D 1-Wasserstein distance (optimal matching on resampled points).
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = REPO_ROOT / "simulation"
PT_DIR = SIM_DIR / "particle_tracking"
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(PT_DIR))

# Import modules directly to avoid simulation/__init__.py side effects.
from config import create_par2_config, update_config_file
from grf_generation import compute_c, fBmGenerator, generate_logk_field
from hydraulic import read_mt3d_file, run_modflow
from par2 import run_par2
from rk45_cpu import particle_tracking_cpu
from utils import fastest_trajectory


def _resolve_executable(env_var, default_path, label):
    env_path = os.environ.get(env_var, None)
    path = Path(env_path).expanduser().resolve() if env_path else Path(default_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"{label} executable not found at '{path}'. "
            f"Set {env_var} or install at {default_path}."
        )
    return path


def _load_par2_debut_trajectories(path_debut, max_traj=5):
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


def _as_path(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    path = np.column_stack((x, y))
    finite = np.all(np.isfinite(path), axis=1)
    return path[finite]


def _resample_polyline(path, n_samples):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("Path must have shape (n, 2).")
    if path.shape[0] == 0:
        raise ValueError("Path is empty.")
    if path.shape[0] == 1:
        return np.repeat(path, n_samples, axis=0)

    seg = np.diff(path, axis=0)
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    arclen = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(arclen[-1])

    if total_len <= 0.0:
        return np.repeat(path[:1], n_samples, axis=0)

    target_s = np.linspace(0.0, total_len, int(n_samples))
    x = np.interp(target_s, arclen, path[:, 0])
    y = np.interp(target_s, arclen, path[:, 1])
    return np.column_stack((x, y))


def _hausdorff_distance(path_a, path_b):
    d_ab = directed_hausdorff(path_a, path_b)[0]
    d_ba = directed_hausdorff(path_b, path_a)[0]
    return float(max(d_ab, d_ba))


def _wasserstein_distance_2d(path_a, path_b, n_samples=256):
    a = _resample_polyline(path_a, n_samples=n_samples)
    b = _resample_polyline(path_b, n_samples=n_samples)
    cost = cdist(a, b, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def _run_one_realization(
    run_id,
    seed,
    args,
    path_root,
    path_mf2005,
    path_par2,
    fbm_gen,
):
    rec = {
        "run_id": int(run_id),
        "seed": int(seed),
        "status": "failed",
        "error": "",
        "modflow_ok": False,
        "par2_ok": False,
        "rk45_time": np.nan,
        "par2_time": np.nan,
        "hausdorff": np.nan,
        "wasserstein": np.nan,
        "rk45_points": 0,
        "par2_points": 0,
    }

    run_dir = path_root / f"mc_{run_id:04d}"
    path_out = run_dir / "compare_output"
    path_mfoutput = run_dir / "mf2005_output"
    path_p2input = run_dir / "par2_input"
    path_p2output = run_dir / "par2_output"
    for p in [path_out, path_mfoutput, path_p2input, path_p2output]:
        p.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    try:
        # --- Domain / fBm parameters ---
        dx = float(args.dx)
        dy = float(args.dy)
        dz = float(args.dz)
        grad_head = float(args.grad_head)

        logk_field, msg_k = generate_logk_field(fbm_gen)
        print(f"[MC {run_id:04d}] {msg_k}")
        if logk_field is None or logk_field.ndim != 2:
            rec["status"] = "invalid_logk"
            return rec

        np.savetxt(path_out / "logk_field.dat", logk_field, delimiter=" ")
        ly, lx = logk_field.shape

        source_vertex = [
            [lx // 20 - 1, ly // 20 - 1, 0],
            [lx // 20 - 1, ly - ly // 20 - 1, 1],
        ]
        target_vertex = [
            [lx - lx // 20 - 1, 0, 0],
            [lx - lx // 20 - 1, ly - 1, 1],
        ]
        x_cp = target_vertex[0][0]

        # --- Shared comparison parameters (same for PAR2 and RK45 in this run) ---
        x0 = float(source_vertex[0][0])
        y0 = float(source_vertex[0][1])
        y1 = float(source_vertex[1][1])
        density_particle = int(args.density_particle)
        nP = max(2, int(density_particle * abs(y1 - y0)))
        dt_compare = 0.5 * min(dx, dy)
        max_steps_compare = 200000
        right_x = float(x_cp)
        eps_speed = 1e-30
        ds_min = 1e-20
        ds_max = 0.5 * min(dx, dy)
        tol = 1e-4
        y_policy = "stop"
        max_points = 50000
        max_iters = 200000
        max_traj = args.max_par2_traj

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
        rec["modflow_ok"] = "successfully" in return_msg_mf.lower()
        if not rec["modflow_ok"]:
            rec["status"] = "modflow_failed"
            rec["error"] = return_msg_mf
            return rec

        # --- PAR2 (if enabled; same nP, dt, steps, and debut num as shared params) ---
        if args.use_par2:
            par2_overrides = {
                "simulation": {
                    "particles": {"N": nP},
                    "dt": dt_compare,
                    "steps": max_steps_compare,
                },
                "output": {"debut": {"num": max_traj}},
            }
            update_config_file(
                str(path_p2input / "par2_config.yaml"),
                create_par2_config(
                    dx, dy, lx, ly, path_mfoutput, path_p2output, source_vertex, x_cp,
                    overrides=par2_overrides,
                ),
            )
            return_code_p2, return_msg_p2 = run_par2(path_par2, path_p2input, path_p2output)
            rec["par2_ok"] = return_code_p2 in (0, 10, 11)
            if not rec["par2_ok"]:
                rec["status"] = "par2_failed"
                rec["error"] = return_msg_p2
                return rec

        # --- Velocity and RK45 CPU ---
        velocity_field = read_mt3d_file(str(ftl_path))
        if "QXX" not in velocity_field or "QYY" not in velocity_field:
            rec["status"] = "missing_velocity"
            rec["error"] = "FTL file missing QXX/QYY."
            return rec

        vx = velocity_field["QXX"]
        vy = velocity_field["QYY"]
        x_grid = np.arange(vx.shape[1], dtype=np.float32) * np.float32(dx)
        y_grid = np.arange(vx.shape[0], dtype=np.float32) * np.float32(dy)
        seeds_x = np.full(nP, x0, dtype=np.float32)
        seeds_y = np.linspace(y0, y1, nP).astype(np.float32)

        args_particle_tracking = {
            "x": x_grid,
            "y": y_grid,
            "vx": vx,
            "vy": vy,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": right_x,
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
        if args.use_rk45_cpu:
            px, py, pt, L, st = particle_tracking_cpu(args_particle_tracking)
            rk45_cpu_res = fastest_trajectory(px, py, pt, L, st)
            rec["rk45_time"] = float(rk45_cpu_res["travel_time"])

        # --- PAR2 fastest (if PAR2 was run) ---
        par2_fast = None
        par2_fast_time = np.nan
        if args.use_par2:
            path_debut = path_p2output / "debut_trajectory"
            _, par2_fast, par2_fast_time = _load_par2_debut_trajectories(path_debut, max_traj=max_traj)
            rec["par2_time"] = float(par2_fast_time) if par2_fast is not None and np.isfinite(par2_fast_time) else np.nan
        if par2_fast is None and args.use_par2:
            rec["status"] = "no_par2_fastest"
            rec["error"] = "No valid PAR2 debut trajectory found."
            return rec

        rk45_path = _as_path(rk45_cpu_res["x"], rk45_cpu_res["y"]) if rk45_cpu_res is not None else np.empty((0, 2), dtype=np.float64)
        par2_path = np.asarray(par2_fast, dtype=np.float64)[np.all(np.isfinite(par2_fast), axis=1)] if par2_fast is not None else np.empty((0, 2), dtype=np.float64)

        rec["rk45_points"] = int(rk45_path.shape[0])
        rec["par2_points"] = int(par2_path.shape[0])

        if rk45_path.shape[0] > 0 and par2_path.shape[0] > 0:
            rec["hausdorff"] = _hausdorff_distance(par2_path, rk45_path)
            rec["wasserstein"] = _wasserstein_distance_2d(
                par2_path,
                rk45_path,
                n_samples=int(args.wasserstein_samples),
            )
        rec["status"] = "ok"

        if args.save_run_data:
            save_kw = {"logk_field": logk_field, "vx": vx, "vy": vy}
            if rk45_path.shape[0] > 0:
                save_kw["rk45_cpu_x"] = rk45_path[:, 0]
                save_kw["rk45_cpu_y"] = rk45_path[:, 1]
                save_kw["rk45_cpu_time"] = rec["rk45_time"]
            if par2_path.shape[0] > 0:
                save_kw["par2_x"] = par2_path[:, 0]
                save_kw["par2_y"] = par2_path[:, 1]
                save_kw["par2_time"] = rec["par2_time"]
            if np.isfinite(rec["hausdorff"]):
                save_kw["hausdorff"] = rec["hausdorff"]
            if np.isfinite(rec["wasserstein"]):
                save_kw["wasserstein"] = rec["wasserstein"]
            np.savez(path_out / "compare_par2_rk45_mc_data.npz", **save_kw)

    except Exception as exc:
        rec["status"] = "exception"
        rec["error"] = str(exc)

    return rec


def _save_records_csv(path_csv, records):
    fieldnames = [
        "run_id",
        "seed",
        "status",
        "error",
        "modflow_ok",
        "par2_ok",
        "rk45_time",
        "par2_time",
        "hausdorff",
        "wasserstein",
        "rk45_points",
        "par2_points",
    ]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _print_run_line(rec, num_runs):
    hid = rec["run_id"] + 1
    h = rec["hausdorff"]
    w = rec["wasserstein"]
    htxt = f"{h:.6g}" if np.isfinite(h) else "nan"
    wtxt = f"{w:.6g}" if np.isfinite(w) else "nan"
    print(f"[MC {hid:04d}/{num_runs:04d}] status={rec['status']}, Hausdorff={htxt}, Wasserstein={wtxt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Monte Carlo comparison: PAR2 vs RK45 fastest paths")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of Monte Carlo realizations")
    parser.add_argument("--seed", type=int, default=547, help="Base random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("compare_mc_output"), help="Output directory")
    parser.add_argument("--num-pt", type=int, default=100, help="Grid size for fBm field generation")
    parser.add_argument("--hurst", type=float, default=0.25, help="Hurst exponent")
    parser.add_argument("--std", type=float, default=2.0, help="Target standard deviation for fBm")
    parser.add_argument("--minpadding", type=int, default=1, help="Minimum fBm padding")
    parser.add_argument("--grad-head", type=float, default=0.1, help="Hydraulic head gradient")
    parser.add_argument("--dx", type=float, default=1.0, help="Cell size in x")
    parser.add_argument("--dy", type=float, default=1.0, help="Cell size in y")
    parser.add_argument("--dz", type=float, default=1.0, help="Cell size in z")
    parser.add_argument("--density-particle", type=int, default=20, help="RK45 seed density per vertical unit")
    parser.add_argument(
        "--wasserstein-samples",
        type=int,
        default=256,
        help="Number of arclength samples per path for Wasserstein distance",
    )
    parser.add_argument("--max-par2-traj", type=int, default=5, help="Max PAR2 debut trajectories to inspect")
    parser.add_argument(
        "--save-run-data",
        action="store_true",
        help="Save per-run npz files with fields and trajectories",
    )
    # Algorithm toggles (same as compare.py); MC only runs PAR2 and RK45 CPU
    parser.add_argument("--no-par2", action="store_true", help="Do not run PAR2 in the comparison.")
    parser.add_argument("--no-rk45-cpu", action="store_true", help="Do not run RK45 CPU in the comparison.")
    parser.add_argument("--no-rk45-gpu", action="store_true", help="Do not run RK45 GPU (ignored in MC).")
    parser.add_argument("--no-rk4-fp", action="store_true", help="Do not run RK4 FP (ignored in MC).")
    args = parser.parse_args()
    args.use_par2 = not args.no_par2
    args.use_rk45_cpu = not args.no_rk45_cpu
    if not (args.use_par2 or args.use_rk45_cpu):
        parser.error("At least one of PAR2 or RK45 CPU must be enabled (do not pass both --no-par2 and --no-rk45-cpu).")
    return args


def main():
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0.")
    if args.wasserstein_samples <= 1:
        raise ValueError("--wasserstein-samples must be > 1.")

    path_cwd = Path(__file__).parent.resolve()
    out_dir = (path_cwd / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    path_mf2005 = _resolve_executable("MODFLOW_EXE", "~/bin/mf2005", "MODFLOW")
    path_par2 = _resolve_executable("PAR2_EXE", "~/bin/par2", "PAR2") if args.use_par2 else None

    alpha = args.hurst * 2.0
    c = compute_c(args.num_pt, args.hurst, args.std)
    fbm_gen = fBmGenerator(c=c, alpha=alpha, num_pt=args.num_pt, dim=2, minpadding=args.minpadding)

    print(f"Running {args.num_runs} Monte Carlo realizations.")
    print(f"Output directory: {out_dir}")
    print(f"MODFLOW executable: {path_mf2005}")
    if args.use_par2:
        print(f"PAR2 executable: {path_par2}")
    if args.use_rk45_cpu:
        print("RK45 CPU: enabled")

    records = []
    for i in range(args.num_runs):
        seed_i = args.seed + i
        rec = _run_one_realization(
            run_id=i,
            seed=seed_i,
            args=args,
            path_root=out_dir,
            path_mf2005=path_mf2005,
            path_par2=path_par2,
            fbm_gen=fbm_gen,
        )
        records.append(rec)
        _print_run_line(rec, args.num_runs)

    path_csv = out_dir / "mc_metrics.csv"
    _save_records_csv(path_csv, records)

    hausdorff_values = [r["hausdorff"] for r in records]
    wasserstein_values = [r["wasserstein"] for r in records]
    summary = {
        "num_runs": int(args.num_runs),
        "num_ok": int(sum(r["status"] == "ok" for r in records)),
        "num_modflow_ok": int(sum(bool(r["modflow_ok"]) for r in records)),
        "num_par2_ok": int(sum(bool(r["par2_ok"]) for r in records)),
        "hausdorff": _summary_stats(hausdorff_values),
        "wasserstein": _summary_stats(wasserstein_values),
    }

    path_json = out_dir / "mc_summary.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "mc_metrics.npz",
        run_id=np.array([r["run_id"] for r in records], dtype=np.int32),
        seed=np.array([r["seed"] for r in records], dtype=np.int64),
        modflow_ok=np.array([r["modflow_ok"] for r in records], dtype=bool),
        par2_ok=np.array([r["par2_ok"] for r in records], dtype=bool),
        rk45_time=np.array([r["rk45_time"] for r in records], dtype=np.float64),
        par2_time=np.array([r["par2_time"] for r in records], dtype=np.float64),
        hausdorff=np.array([r["hausdorff"] for r in records], dtype=np.float64),
        wasserstein=np.array([r["wasserstein"] for r in records], dtype=np.float64),
        rk45_points=np.array([r["rk45_points"] for r in records], dtype=np.int32),
        par2_points=np.array([r["par2_points"] for r in records], dtype=np.int32),
        status=np.array([r["status"] for r in records], dtype=str),
    )

    print("\nMonte Carlo summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved per-run metrics: {path_csv}")
    print(f"Saved summary: {path_json}")
    print(f"Saved arrays: {out_dir / 'mc_metrics.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
