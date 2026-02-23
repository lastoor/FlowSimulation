#!/usr/bin/env python
"""
Hybrid Monte Carlo runner using simulation package APIs.

Workflow per realization:
1) fBm log-K generation
2) PETSc head solve
3) Face-centered velocity
4) Least-resistance path (Dijkstra)
5) Fastest path via GPU RK4 (Taichi), with CPU RK45 fallback
"""

import argparse
import cProfile
import gc
import multiprocessing as mp
import os
import pstats
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# Keep runtime library path behavior consistent with existing scripts.
os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Ensure simulation package import when invoked from scripts/.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import config as sim_config
from simulation import grf_generation
from simulation import hydraulic
from simulation import path_finding
from simulation import utils
from simulation import particle_tracking


global_queue_gpu = None
global_result_dict = None
global_fbm_gen = None


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid CPU+GPU Monte Carlo flow simulation runner")
    parser.add_argument("--path_workdir", required=True, type=str)
    parser.add_argument("--num_mc", required=True, nargs="+", type=int)
    parser.add_argument("--field_size", required=True, nargs="+", type=int)
    parser.add_argument("--hurst", required=True, type=float)
    parser.add_argument("--minpadding", type=int, default=0)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--cpu_workers", type=int, default=None)
    parser.add_argument("--gpu_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_trial", type=int, default=50)
    return parser.parse_args()


def init_worker(queue_gpu, result_dict, fbm_gen):
    global global_queue_gpu, global_result_dict, global_fbm_gen
    global_queue_gpu = queue_gpu
    global_result_dict = result_dict
    global_fbm_gen = fbm_gen


def parse_visible_gpu_tokens():
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        tokens = [t.strip() for t in env.split(",") if t.strip()]
        tokens = [t for t in tokens if t != "-1"]
        if tokens:
            return tokens

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU ")]
            if lines:
                return [str(i) for i in range(len(lines))]
    except Exception:
        pass

    return []


def compute_travel_time_lrp(lrp, vx, vy, x, y, eps_speed=1e-12):
    x_phys = x[lrp[:, 0]]
    y_phys = y[lrp[:, 1]]

    total_time = 0.0
    for i in range(len(lrp) - 1):
        dx = x_phys[i + 1] - x_phys[i]
        dy = y_phys[i + 1] - y_phys[i]
        ds = np.sqrt(dx * dx + dy * dy)
        if ds <= 0:
            continue

        vx_i = vx[lrp[i, 1], lrp[i, 0]]
        vy_i = vy[lrp[i, 1], lrp[i, 0]]
        v_proj = (vx_i * dx + vy_i * dy) / ds
        if v_proj <= 0:
            v_proj = eps_speed
        total_time += ds / v_proj

    return float(total_time)


def generate_logk_field(_config_mc):
    return grf_generation.generate_logk_field(global_fbm_gen)


def gpu_worker(config_base, gpu_token, event_stop, queue_gpu, result_dict):
    logger = utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info(f"[GPU-Worker {gpu_token}] Starting")

    # Pin this process to a single GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token

    while not event_stop.is_set() or not queue_gpu.empty():
        try:
            task = queue_gpu.get(timeout=0.5)
        except Exception:
            continue

        task_id = task.get("id")
        func_name = task.get("func")
        args = task.get("args", tuple())
        kwargs = task.get("kwargs", {})

        result = None
        if func_name == "rk4_fastest_path":
            try:
                from simulation.particle_tracking.rk4_fp import rk4_fastest_path_params
                res = rk4_fastest_path_params(*args, **kwargs)
                result = {"ok": True, "result": res, "error": ""}
            except Exception as e:
                result = {"ok": False, "result": None, "error": str(e)}
        else:
            result = {"ok": False, "result": None, "error": f"Unknown func '{func_name}'"}

        result_dict[task_id] = result

    logger.info(f"[GPU-Worker {gpu_token}] Stopping")


def gpu_call(func_name, payload, timeout_sec=7200):
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "func": func_name,
        "args": (payload,),
        "kwargs": {},
    }

    global_queue_gpu.put(task)

    t0 = time.time()
    while task_id not in global_result_dict:
        if (time.time() - t0) > timeout_sec:
            raise TimeoutError(f"GPU task timeout for {func_name} after {timeout_sec}s")
        time.sleep(0.01)

    return global_result_dict.pop(task_id)


def run_unit_simulation(config, num_mc_, num_trial=50, seed=None):
    logger = utils.get_logger(path_logfile=str(config["path_mainlogfile"]))
    logger.info(f"[MC {num_mc_}] Starting simulation (up to {num_trial} trials)")

    config_mc = config.copy()
    config_mc["path_mc"] = config["path_size"] / f"mc{num_mc_:04d}"

    if seed is not None:
        np.random.seed(seed)

    x0 = config_mc["source_vertex"][0][0]
    x1 = config_mc["target_vertex"][1][0]
    y0 = config_mc["source_vertex"][0][1]
    y1 = config_mc["source_vertex"][1][1]
    domain_box = (x0, x1, y0, y1)

    for trial in range(num_trial):
        logger.info(f"[MC {num_mc_}] Trial {trial + 1}/{num_trial}")

        logk_field, return_msg_k = generate_logk_field(config_mc)
        if logk_field is None:
            logger.warning(f"[MC {num_mc_}] {return_msg_k}")
            continue
        np.save(
            config_mc["path_size_logk"] / f"logKField_size{config_mc['field_size_']}_mc{num_mc_:04d}.npy",
            logk_field,
        )
        k_field = np.exp(logk_field)

        logger.info(f"[MC {num_mc_}] {return_msg_k}")

        source = [(y, x0) for y in range(y0, y1)]
        target = [(y, x1) for y in range(0, config_mc["ly"] - 1)]
        lrp, total_R, _s_node, _e_node, _rmap = path_finding.dijkstra_2d_least_resistance(
            k_field, source, target, average="harmonic"
        )

        if lrp.size == 0 or not path_finding.is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid LRP, retrying")
            continue

        h, msg, is_successful = hydraulic.solve_head(
            k_field,
            config_mc["x"],
            config_mc["y"],
            dh=config_mc["headSolver_arg"]["dh"],
            rtol=config_mc["headSolver_arg"]["rtol"],
            atol=config_mc["headSolver_arg"]["atol"],
            max_it=config_mc["headSolver_arg"]["max_it"],
        )
        if not is_successful:
            logger.warning(f"[MC {num_mc_}] Head solve failed: {msg}")
            continue

        vx, vy = hydraulic.compute_velocity(
            k_field,
            config_mc["x"],
            config_mc["y"],
            h,
            output_location="face",
        )
        t_lrp = compute_travel_time_lrp(
            lrp,
            vx,
            vy,
            config_mc["x"],
            config_mc["y"],
            eps_speed=config_mc["particleTracking_arg"]["eps_speed"],
        )

        nP = int(config_mc["particleTracking_arg"]["density_particle"] * np.abs(y1 - y0))
        nP = max(2, nP)
        seeds_x = np.full(nP, x0, np.float32)
        seeds_y = np.linspace(y0, y1, nP).astype(np.float32)

        args_tracking = {
            "x": config_mc["x"],
            "y": config_mc["y"],
            "vx": vx,
            "vy": vy,
            "seeds_x": seeds_x,
            "seeds_y": seeds_y,
            "right_x": x1,
            "eps_speed": config_mc["particleTracking_arg"]["eps_speed"],
            "ds_min": config_mc["particleTracking_arg"]["ds_min"],
            "ds_max": config_mc["particleTracking_arg"]["ds_max"],
            "tol": config_mc["particleTracking_arg"]["tol"],
            "y_policy": config_mc["particleTracking_arg"]["y_policy"],
            "max_points": config_mc["particleTracking_arg"]["max_points"],
            "max_iters": config_mc["particleTracking_arg"]["max_iters"],
            "dtype": np.float64,
        }

        # GPU RK4 first, then CPU RK45 fallback.
        res = None
        fp = None
        travel_time = np.nan

        if config_mc.get("use_gpu_rk4", False):
            rk4_payload = {
                "x": config_mc["x"],
                "y": config_mc["y"],
                "vx": vx,
                "vy": vy,
                "seeds_x": seeds_x,
                "seeds_y": seeds_y,
                "right_x": x1,
                "dt": config_mc["rk4Tracking_arg"]["dt"],
                "max_steps": config_mc["rk4Tracking_arg"]["max_steps"],
                "y_policy": config_mc["particleTracking_arg"]["y_policy"],
                "closed_boundary": config_mc["rk4Tracking_arg"]["closed_boundary"],
            }
            try:
                rk4_out = gpu_call("rk4_fastest_path", rk4_payload)
                if rk4_out.get("ok", False):
                    r = rk4_out["result"]
                    fp = np.vstack((r["x"], r["y"])).T
                    travel_time = float(r["travel_time"])
                    res = {"travel_time": travel_time}
                else:
                    logger.warning(f"[MC {num_mc_}] RK4 failed on GPU: {rk4_out.get('error', 'unknown')}")
            except Exception as e:
                logger.warning(f"[MC {num_mc_}] RK4 GPU task failed: {e}")

        if fp is None:
            try:
                px, py, pt, L, st = particle_tracking.particle_tracking_cpu(args_tracking)
                count = Counter(st.tolist())
                if count[0] / nP < 0.5 or (count[1] + count[4]) / nP > 0.1:
                    logger.warning(f"[MC {num_mc_}] Invalid velocity field for CPU RK45 fallback: {count}")
                    continue
                r = particle_tracking.fastest_trajectory(px, py, pt, L, st)
                fp = np.vstack((r["x"], r["y"])).T
                travel_time = float(r["travel_time"])
                res = {"travel_time": travel_time}
            except Exception as e:
                logger.warning(f"[MC {num_mc_}] CPU RK45 fallback failed: {e}")
                continue

        fp = fp[fp[:, 0] <= config_mc["x_cp"]]
        if fp.size == 0:
            logger.warning(f"[MC {num_mc_}] Empty FP after trimming")
            continue

        if not path_finding.is_path_valid(fp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP geometry, retrying")
            continue

        len_lrp = utils.euclidean_length(lrp)
        len_fp = utils.euclidean_length(fp)

        main_output_m = {
            "len_lrp_m": len_lrp,
            "len_fp_m": len_fp,
            "fpt_m": res["travel_time"],
            "mhr_m": total_R,
            "t_lrp_m": t_lrp,
        }
        np.savez(
            str(config["path_size_mainOutput"] / f"mainOutput_size{config_mc['field_size_']}_mc{num_mc_:04d}.npz"),
            **main_output_m,
        )

        np.save(config_mc["path_size_path"] / f"lrp_size{config_mc['field_size_']}_mc{num_mc_:04d}.npy", lrp)
        np.save(config_mc["path_size_path"] / f"fp_size{config_mc['field_size_']}_mc{num_mc_:04d}.npy", fp)

        logger.info(f"[MC {num_mc_}] Success on trial {trial + 1}")
        return len_lrp, len_fp, res["travel_time"], total_R, t_lrp

    logger.error(f"[MC {num_mc_}] All {num_trial} trials failed")
    return np.nan, np.nan, np.nan, np.nan, np.nan


def run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict, num_trial=50, seed=None):
    field_size_ = config_base["field_size"][idx_size]
    logger = utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info(f"[SIZE {field_size_}] Starting")

    config = config_base.copy()
    config.update(
        {
            "idx_size": idx_size,
            "field_size_": field_size_,
            "lx": int(field_size_),
            "ly": int(field_size_),
            "lz": 1,
        }
    )
    config.update(
        {
            "source_vertex": [
                [config["lx"] // 20 - 1, config["ly"] // 20 - 1, 0],
                [config["lx"] // 20 - 1, config["ly"] - config["ly"] // 20 - 1, 1],
            ],
            "target_vertex": [
                [config["lx"] - config["lx"] // 20 - 1, 0, 0],
                [config["lx"] - config["lx"] // 20 - 1, config["ly"] - 1, 1],
            ],
        }
    )
    config["x_cp"] = config["target_vertex"][0][0]
    config["x"] = np.arange(0, field_size_, config["dx"])
    config["y"] = np.arange(0, field_size_, config["dy"])

    config["path_size"] = config["path_simulation"] / f"hurst{config['fbm_arg']['hurst']}_size{field_size_}"
    config["path_size_mainOutput"] = config["path_size"] / "main_output"
    config["path_size_path"] = config["path_size"] / "path"
    config["path_size_logk"] = config["path_size"] / "logk_field"

    if config["path_size"].exists() and config["path_size"].is_dir():
        shutil.rmtree(config["path_size"])
    config["path_size"].mkdir(parents=True, exist_ok=True)
    config["path_size_mainOutput"].mkdir(parents=True, exist_ok=True)
    config["path_size_path"].mkdir(parents=True, exist_ok=True)
    config["path_size_logk"].mkdir(parents=True, exist_ok=True)

    if config["fbm_arg"]["std"] is not None:
        c = grf_generation.compute_c(field_size_, config["fbm_arg"]["hurst"], config["fbm_arg"]["std"])
    else:
        c = config["fbm_arg"]["c"]

    fbm_gen = grf_generation.fBmGenerator(
        c=c,
        alpha=config["fbm_arg"]["hurst"] * 2,
        num_pt=field_size_,
        minpadding=config["fbm_arg"]["minpadding"],
    )

    num_mc_ = config["num_mc"][idx_size]
    if config_base["cpu_workers"] is None:
        cpu_workers = min(num_mc_, len(os.sched_getaffinity(0)))
    else:
        cpu_workers = max(1, min(num_mc_, config_base["cpu_workers"]))

    seeds_for_mc = None
    if seed is not None:
        rng = np.random.default_rng(seed + idx_size * 100000)
        seeds_for_mc = [int(v) for v in rng.integers(0, 2**31 - 1, size=num_mc_)]

    with mp.Pool(processes=cpu_workers, initializer=init_worker, initargs=(queue_gpu, result_dict, fbm_gen)) as pool:
        jobs = []
        for i in range(num_mc_):
            mc_seed = None if seeds_for_mc is None else seeds_for_mc[i]
            jobs.append((config, i, num_trial, mc_seed))
        mc_result = pool.starmap(run_unit_simulation, jobs)

    len_lrp_sm = np.array([r[0] for r in mc_result]) if mc_result else np.array([])
    len_fp_sm = np.array([r[1] for r in mc_result]) if mc_result else np.array([])
    fpt_sm = np.array([r[2] for r in mc_result]) if mc_result else np.array([])
    mhr_sm = np.array([r[3] for r in mc_result]) if mc_result else np.array([])
    t_lrp_sm = np.array([r[4] for r in mc_result]) if mc_result else np.array([])

    main_output = {
        "len_lrp_sm": len_lrp_sm,
        "len_fp_sm": len_fp_sm,
        "fpt_sm": fpt_sm,
        "mhr_sm": mhr_sm,
        "t_lrp_sm": t_lrp_sm,
    }
    np.savez(str(config["path_size_mainOutput"] / f"mainOutput_size{field_size_}.npz"), **main_output)

    logger.info(f"[SIZE {field_size_}] Completed")
    return idx_size, len_lrp_sm, len_fp_sm


def main():
    args = parse_args()

    if len(args.num_mc) != len(args.field_size):
        raise ValueError("--num_mc and --field_size must have the same length")
    if (args.c is None) == (args.std is None):
        raise ValueError("Provide exactly one of --c or --std")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Reuse base config shape from simulation.config helpers.
    cfg_args = argparse.Namespace(
        path_workdir=args.path_workdir,
        num_mc=args.num_mc,
        field_size=args.field_size,
        hurst=args.hurst,
        minpadding=args.minpadding,
        c=args.c,
        std=args.std,
    )
    config_base = sim_config.create_base_config(cfg_args)

    config_base["path_mainlogfile"] = config_base["path_log"] / f"simulation_{timestamp}.log"
    config_base["path_gpulogfile"] = config_base["path_log"] / f"gpu_profile_{timestamp}.log"
    config_base["path_cpulogfile"] = config_base["path_log"] / f"cpu_profile_{timestamp}.log"

    config_base["path_simulation"].mkdir(parents=True, exist_ok=True)
    config_base["path_log"].mkdir(parents=True, exist_ok=True)

    # Extra runtime knobs for this hybrid script.
    config_base["cpu_workers"] = args.cpu_workers
    config_base["rk4Tracking_arg"] = {
        "dt": 0.1 * min(config_base["dx"], config_base["dy"]),
        "max_steps": 200000,
        "closed_boundary": True,
    }

    gpu_tokens = parse_visible_gpu_tokens()
    if args.gpu_workers is not None:
        gpu_tokens = gpu_tokens[: max(0, args.gpu_workers)]
    config_base["use_gpu_rk4"] = len(gpu_tokens) > 0

    logger = utils.get_logger(path_logfile=str(config_base["path_mainlogfile"]))
    logger.info("[MAIN] Starting hybrid simulation")
    logger.info(f"[MAIN] GPU tokens: {gpu_tokens}")

    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(
        target=utils.log_gpu_usage,
        kwargs={
            "path_logfile": str(config_base["path_gpulogfile"]),
            "interval": 1.0,
            "event_stop": event_stop_gpuprofiler,
        },
    )
    thread_gpuprofiler.start()

    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    manager = mp.Manager()
    result_dict = manager.dict()
    queue_gpu = mp.Queue()

    event_stop_gpuworkers = mp.Event()
    gpu_processes = []
    for token in gpu_tokens:
        p = mp.Process(
            target=gpu_worker,
            args=(config_base, token, event_stop_gpuworkers, queue_gpu, result_dict),
            name=f"GPU-Worker-{token}",
        )
        p.start()
        gpu_processes.append(p)

    try:
        for idx_size in range(len(config_base["field_size"])):
            run_field_size_simulation(
                idx_size=idx_size,
                config_base=config_base,
                queue_gpu=queue_gpu,
                result_dict=result_dict,
                num_trial=args.num_trial,
                seed=args.seed,
            )
    finally:
        event_stop_gpuworkers.set()
        for p in gpu_processes:
            p.join(timeout=20)

    cpu_profiler.disable()
    with open(str(config_base["path_cpulogfile"]), "w") as f:
        pstats.Stats(cpu_profiler, stream=f).sort_stats("time").print_stats()

    event_stop_gpuprofiler.set()
    thread_gpuprofiler.join(timeout=10)

    end_time = datetime.now()
    elapsed_hours = round((end_time - start_time).total_seconds() / 3600.0, 2)
    logger.info(
        f"[MAIN] Completed simulation in {elapsed_hours} hours "
        f"from {start_time:%Y-%m-%d %H:%M:%S} to {end_time:%Y-%m-%d %H:%M:%S}"
    )

    gc.collect()


if __name__ == "__main__":
    main()
