"""
Main simulation orchestration.
"""

import os
import sys
import time
import gc
import threading
import cProfile
import pstats
import uuid
import warnings
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import multiprocessing as mp
import numpy as np

# Local imports
from . import config
from . import grf_generation
from . import hydraulic
from . import particle_tracking
from . import path_finding
from . import utils

# Optional GPU imports
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    _cupy_available = False

# Global variables for multiprocessing (set by init_worker)
global_queue_gpu = None
global_result_dict = None
global_fbm_gen = None


def init_worker(queue_gpu, result_dict, fbm_gen):
    """Initializer for simulation pool workers.
       Sets the shared GPU task queue and result dictionary as globals."""
    global global_queue_gpu, global_result_dict, global_fbm_gen
    global_queue_gpu = queue_gpu
    global_result_dict = result_dict
    global_fbm_gen = fbm_gen


def generate_logk_field(config_mc):
    """Generate the logk field based on the chosen method."""
    start_time_fbm = time.time()
    try:
        logk_field = global_fbm_gen.sample()
    except Exception as e:
        elapsed_time_fbm = time.time() - start_time_fbm
        return_msg = f'fBm field generated unsuccessfully {np.round(elapsed_time_fbm, 4)} seconds: {e}'
        return None, return_msg

    elapsed_time_fbm = time.time() - start_time_fbm
    return_msg = f'fBm field generated successfully after {np.round(elapsed_time_fbm, 4)} seconds.'
    return logk_field, return_msg


def compute_travel_time_lrp(lrp, vx, vy, x, y, eps_speed=1e-12):
    """
    Compute travel time along LRP by integrating ds/v_lrp.

    Parameters:
        lrp (np.ndarray): Path array of shape (n,2) with (col, row) indices.
        vx, vy (np.ndarray): Velocity components shape (ny, nx).
        x, y (np.ndarray): Physical coordinates of cell centers.
        eps_speed (float): Minimum velocity to avoid division by zero.

    Returns:
        travel_time (float): Total travel time along LRP.
    """
    # Convert indices to physical coordinates
    x_phys = x[lrp[:,0]]
    y_phys = y[lrp[:,1]]

    total_time = 0.0
    for i in range(len(lrp)-1):
        dx = x_phys[i+1] - x_phys[i]
        dy = y_phys[i+1] - y_phys[i]
        ds = np.sqrt(dx*dx + dy*dy)
        if ds <= 0:
            continue
        # Velocity at start point
        vx_i = vx[lrp[i,1], lrp[i,0]]
        vy_i = vy[lrp[i,1], lrp[i,0]]
        # Projection of velocity onto segment direction
        v_proj = (vx_i*dx + vy_i*dy) / ds
        if v_proj <= 0:
            v_proj = eps_speed
        dt = ds / v_proj
        total_time += dt
    return total_time


def _build_faces_from_centers(centers):
    centers = np.asarray(centers, dtype=np.float32).ravel()
    if centers.size < 2:
        raise ValueError("center coordinates must contain at least 2 values.")
    faces = np.empty(centers.size + 1, dtype=np.float32)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    faces[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    faces[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return faces


def gpu_worker(config, event_stop_gpuworker, queue_gpu, result_dict):
    """
    GPU worker process: pulls tasks from the GPU task queue, processes them,
    and writes results into the shared result dictionary.
    """
    logger = utils.get_logger(path_logfile=str(config['path_mainlogfile']))
    logger.info('[GPU-Worker] Starting with 1 GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 only.
    while not event_stop_gpuworker.is_set() or not queue_gpu.empty():
        try:
            task = queue_gpu.get(timeout=0.5)
        except Exception:
            continue
        task_id = task.get('id')
        func_name = task.get('func')
        args = task.get('args', tuple())
        kwargs = task.get('kwargs', {})
        log_msg = task.get('log_msg')

        # Dispatch GPU tasks
        logger.info(f"[GPU-Worker] Starting task '{log_msg}'.")
        if func_name == 'generate_logk_field':
            result = generate_logk_field(*args, **kwargs)
        elif func_name == 'run_par2':
            # This function is not defined in this package; you may need to implement or import
            logger.warning('[GPU-Worker] run_par2 not implemented.')
            result = None
        elif func_name == 'particle_tracking_rk45':
            # This function is not defined in this package; you may need to implement or import
            logger.warning('[GPU-Worker] particle_tracking_rk45 not implemented.')
            result = None
        else:
            logger.info('[GPU-Worker] Unmatched `func_name`. Please choose either `generate_logk_field` or `run_par2`.')
            result = None
        # Put the result into the shared dictionary with task_id as key.
        result_dict[task_id] = result
    logger.info('[GPU-Worker] Shutting down.')


def gpu_call(func_name, config, log_msg):
    task_id = uuid.uuid4().hex  # Unique task id
    task = {
        'id': task_id,
        'func': func_name,
        'args': (config,),  # pass config as argument
        'kwargs': {},
        'log_msg': log_msg
    }
    global_queue_gpu.put(task)  # Submit task to the GPU queue

    # Poll until the result arrives in the shared dictionary.
    while task_id not in global_result_dict:
        time.sleep(0.01)

    # Retrieve and remove the result.
    return global_result_dict.pop(task_id)


def run_unit_simulation(config, num_mc_, num_trial=50):
    """
    Run a single Monte Carlo simulation with controlled retry logic.
    """
    logger = utils.get_logger(path_logfile=str(config['path_mainlogfile']))
    logger.info(f"[MC {num_mc_}] Starting simulation (up to {num_trial} trials).")
    t_lrp = np.nan

    config_mc = config.copy()
    config_mc['path_mc'] = config['path_size'] / f"mc{num_mc_}"

    # Domain limits
    x0 = config_mc['source_vertex'][0][0]
    x1 = config_mc['target_vertex'][1][0]
    y0 = config_mc['source_vertex'][0][1]
    y1 = config_mc['source_vertex'][1][1]
    domain_box = (x0, x1, y0, y1)

    # Try multiple realizations if needed
    for trial in range(num_trial):
        logger.info(f"[MC {num_mc_}] Trial {trial + 1}/{num_trial} starting.")

        # === 1. Generate random log-conductivity field ===
        logk_field, return_msg_k = generate_logk_field(config_mc)
        np.save(config_mc['path_size_logk'] / f"logKField_size{config_mc['field_size_']}_mc{num_mc_}.npy", logk_field)
        k_field = np.exp(logk_field)
        logger.info(f"[MC {num_mc_}] {return_msg_k}")

        # === 2. Compute LRP ===
        logger.info(f"[MC {num_mc_}] Computing LRP.")

        source = [(y, x0) for y in range(y0, y1)]
        target = [(y, x1) for y in range(0, config_mc['ly']-1)]
        lrp, total_R, s_node, e_node, Rmap = path_finding.dijkstra_2d_least_resistance(
            k_field, source, target, average='harmonic'
        )
        len_lrp = utils.euclidean_length(lrp)

        # Check LRP geometry
        if not path_finding.is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid LRP, retrying next trial.")
            continue

        # === 3. Compute FP ===
        logger.info(f"[MC {num_mc_}] Computing FP.")
        h, msg, is_successful = hydraulic.solve_head(
            k_field, config_mc['x'], config_mc['y'],
            dh=config_mc['headSolver_arg']['dh'],
            rtol=config_mc['headSolver_arg']['rtol'],
            atol=config_mc['headSolver_arg']['atol'],
            max_it=config_mc['headSolver_arg']['max_it']
        )
        if not is_successful:
            logger.warning(f"[MC {num_mc_}] Head not converged, retrying next trial. Error: {msg}")
            continue
        # Velocity field
        vx, vy = hydraulic.compute_velocity(k_field, config_mc['x'], config_mc['y'], h)
        tracking_method = config_mc['particleTracking_arg'].get('method', 'rk45')
        if tracking_method == 'pollock':
            vx_pollock, vy_pollock = hydraulic.compute_velocity(
                k_field,
                config_mc['x'],
                config_mc['y'],
                h,
                output_location='face',
                face_output='full',
            )
        else:
            vx_pollock = vy_pollock = None
        # Compute travel time along LRP
        t_lrp = compute_travel_time_lrp(lrp, vx, vy, config_mc['x'], config_mc['y'], eps_speed=config_mc['particleTracking_arg']['eps_speed'])

        nP = int(config_mc['particleTracking_arg']['density_particle'] * np.abs(y1 - y0))
        seeds_x = np.full(nP, x0, np.float32)
        seeds_y = np.linspace(y0, y1, nP).astype(np.float32)

        # Particle tracking
        args_particleTracking = {
            'x': config_mc['x'],
            'y': config_mc['y'],
            'vx': vx,
            'vy': vy,
            'seeds_x': seeds_x,
            'seeds_y': seeds_y,
            'backend': config_mc['particleTracking_arg'].get('backend', 'cpu'),
            'method': tracking_method,
            'tracking_mode': config_mc['particleTracking_arg'].get('tracking_mode', 'trajectory'),
            'stop_mode': config_mc['particleTracking_arg'].get('stop_mode', 'first'),
            'arrival_fraction': config_mc['particleTracking_arg'].get('arrival_fraction', 1.0),
            'save_trajectories': config_mc['particleTracking_arg'].get('save_trajectories', True),
            'return_fastest': config_mc['particleTracking_arg'].get('return_fastest', True),
            'right_x': x1,
            'eps_speed': config_mc['particleTracking_arg']['eps_speed'],
            'ds_min': config_mc['particleTracking_arg']['ds_min'],
            'ds_max': config_mc['particleTracking_arg']['ds_max'],
            'tol': config_mc['particleTracking_arg']['tol'],
            'y_policy': config_mc['particleTracking_arg']['y_policy'],
            'max_points': config_mc['particleTracking_arg']['max_points'],
            'max_iters': config_mc['particleTracking_arg']['max_iters'],
            'threads_per_block': config_mc['particleTracking_arg']['threads_per_block'],
            'dtype': np.float64
        }
        if tracking_method == 'pollock':
            args_particleTracking.update(
                {
                    'x_faces': _build_faces_from_centers(config_mc['x']),
                    'y_faces': _build_faces_from_centers(config_mc['y']),
                    'vx': vx_pollock,
                    'vy': vy_pollock,
                    'max_steps': config_mc['particleTracking_arg']['max_iters'],
                }
            )

        tracking_result = particle_tracking.particle_tracking_summary(args_particleTracking)
        st = tracking_result['status']

        # 0=success, 1=stagnation, 4=other errors
        from collections import Counter
        count = Counter(st.tolist())
        if count[0]/nP < 0.5 or (count[1] + count[4])/nP > 0.1:
            logger.warning(f"[MC {num_mc_}] Invalid velocity field, retrying next trial. Error: {count}")
            continue

        # Fastest path
        res = tracking_result.get('fastest')
        if res is None:
            raise ValueError("Current simulation flow requires a fastest trajectory; set return_fastest=True.")
        fp = np.vstack((res["x"], res["y"])).T
        fp = fp[fp[:,0] <= config_mc['x_cp']] # Trim the traj
        len_fp = utils.euclidean_length(fp)

        # Check FP geometry
        if not path_finding.is_path_valid(fp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP, retrying next trial.")
            continue

        # === 4. Save and exit successfully ===
        mainOutput_m = {
            'len_lrp_m': len_lrp,
            'len_fp_m': len_fp,
            'fpt_m': res['travel_time'],
            'mhr_m': total_R,
            't_lrp_m': t_lrp
        }
        np.savez(str(config['path_size_mainOutput'] / f"mainOutput_size{config_mc['field_size_']}_mc{num_mc_}.npz"), **mainOutput_m)

        np.save(config_mc['path_size_path'] / f"lrp_size{config_mc['field_size_']}_mc{num_mc_}.npy", lrp)
        np.save(config_mc['path_size_path'] / f"fp_size{config_mc['field_size_']}_mc{num_mc_}.npy", fp)
        logger.info(f"[MC {num_mc_}] Paths saved.")

        logger.info(f"[MC {num_mc_}] Trial {trial + 1} successful.")
        break

    else:
        len_lrp = np.nan
        len_fp = np.nan
        travel_time = np.nan
        total_R = np.nan
        t_lrp = np.nan
        logger.error(f"[MC {num_mc_}] All {num_trial} trials failed to produce valid output.")
        return len_lrp, len_fp, travel_time, total_R, t_lrp

    return len_lrp, len_fp, res['travel_time'], total_R, t_lrp


def run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict):
    """Run simulations for a given field size with parallelized Monte Carlo runs."""

    # Update config for this field size
    field_size_ = config_base['field_size'][idx_size]
    logger = utils.get_logger(path_logfile=str(config_base['path_mainlogfile']))
    logger.info(f"[SIZE {field_size_}] Starting simulation.")
    config = config_base.copy()
    config.update({
        'idx_size': idx_size,
        'field_size_': field_size_,
        'lx': int(field_size_), 'ly': int(field_size_), 'lz': 1,
    })
    config.update({
        'source_vertex': [[config['lx']//20 - 1, config['ly']//20 - 1, 0],
                          [config['lx']//20 - 1, config['ly'] - config['ly']//20 - 1, 1]],
        'target_vertex': [[config['lx'] - config['lx']//20 - 1, 0, 0],
                          [config['lx'] - config['lx']//20 - 1, config['ly'] - 1, 1]]
    })
    config['x_cp'] = config['target_vertex'][0][0]
    config['x'] = np.arange(0, field_size_, config['dx'])
    config['y'] = np.arange(0, field_size_, config['dy'])

    config['path_size'] = config['path_simulation'] / f"hurst{config['fbm_arg']['hurst']}_size{field_size_}"
    config['path_size_mainOutput'] = config['path_size'] / 'main_output'
    config['path_size_path'] = config['path_size'] / 'path'
    config['path_size_logk'] = config['path_size'] / 'logk_field'

    if config['path_size'].exists() and config['path_size'].is_dir():
        shutil.rmtree(config['path_size'])
    config['path_size'].mkdir(parents=True, exist_ok=True)
    config['path_size_mainOutput'].mkdir(parents=True, exist_ok=True)
    config['path_size_path'].mkdir(parents=True, exist_ok=True)
    config['path_size_logk'].mkdir(parents=True, exist_ok=True)

    # Initialize a fBm generator
    if config['fbm_arg']['std'] is not None:
        c = grf_generation.compute_c(field_size_, config['fbm_arg']['hurst'], config['fbm_arg']['std'])
    elif config['fbm_arg']['c'] is not None:
        c = config['fbm_arg']['c']

    fbm_gen = grf_generation.fBmGenerator(
        c=c,
        alpha=config['fbm_arg']['hurst']*2,
        num_pt=field_size_,
        minpadding=config['fbm_arg']['minpadding']
    )

    # Run Monte Carlo simulations in parallel
    print(config['num_mc'], idx_size)
    num_mc_ = config['num_mc'][idx_size]
    num_process = min(num_mc_, len(os.sched_getaffinity(0)))
    with mp.Pool(processes=num_process, initializer=init_worker, initargs=(queue_gpu, result_dict, fbm_gen)) as pool:
        mc_result = pool.starmap(run_unit_simulation, [(config, i) for i in range(num_mc_)])

    # Collect and save length of paths
    # mc_result is list of tuples: (len_lrp, len_fp, travel_time, total_R, t_lrp)
    # Transpose to get arrays for each quantity
    if mc_result:
        len_lrp_sm = np.array([r[0] for r in mc_result])
        len_fp_sm = np.array([r[1] for r in mc_result])
        fpt_sm = np.array([r[2] for r in mc_result])
        mhr_sm = np.array([r[3] for r in mc_result])
        t_lrp_sm = np.array([r[4] for r in mc_result])
    else:
        len_lrp_sm = np.array([])
        len_fp_sm = np.array([])
        fpt_sm = np.array([])
        mhr_sm = np.array([])
        t_lrp_sm = np.array([])
    mainOutput = {
        'len_lrp_sm': len_lrp_sm,
        'len_fp_sm': len_fp_sm,
        'fpt_sm': fpt_sm,
        'mhr_sm': mhr_sm,
        't_lrp_sm': t_lrp_sm
    }
    np.savez(str(config['path_size_mainOutput'] / f'mainOutput_size{field_size_}.npz'), **mainOutput)
    logger.info(f"[SIZE {field_size_}] Main outputs saved.")
    logger.info(f"[SIZE {field_size_}] Completed simulation.")

    return idx_size, len_lrp_sm, len_fp_sm


def main():
    # Parse arguments from command line
    args = config.parse_args()
    print("\nAll input parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

    mp.set_start_method('spawn')

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    path_workdir = Path(args.path_workdir).expanduser().resolve()

    # Base configuration dictionary
    config_base = config.create_base_config(args)
    # Update log file names with timestamp
    config_base['path_mainlogfile'] = config_base['path_log'] / f'simulation_{timestamp}.log'
    config_base['path_gpulogfile'] = config_base['path_log'] / f'gpu_profile_{timestamp}.log'
    config_base['path_cpulogfile'] = config_base['path_log'] / f'cpu_profile_{timestamp}.log'

    # Ensure output directory exists
    config_base['path_simulation'].mkdir(parents=True, exist_ok=True)
    config_base['path_log'].mkdir(parents=True, exist_ok=True)

    # Setup main logger
    logger = utils.get_logger(path_logfile=str(config_base['path_mainlogfile']))
    logger.info(f"[MAIN] Starting simulation.")

    # Setup GPU profiling
    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(
        target=utils.log_gpu_usage,
        kwargs={
            'path_logfile': str(config_base['path_gpulogfile']),
            'interval': 1.0,
            'event_stop': event_stop_gpuprofiler
        }
    )
    thread_gpuprofiler.start()

    # Setup CPU profiling
    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    # Setup GPU Worker process
    manager = mp.Manager()
    result_dict = manager.dict()
    queue_gpu = mp.Queue()
    event_stop_gpuworker = mp.Event()  # Separate event for the GPU worker process
    process_gpu_worker = mp.Process(
        target=gpu_worker,
        args=(config_base, event_stop_gpuworker, queue_gpu, result_dict),
        name="GPU-Worker"
    )
    process_gpu_worker.start()

    # Run simulations for each field size sequentially
    print(config_base['field_size'])
    for idx_size in range(len(config_base['field_size'])):
        run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict)

    # End of simulation
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    elapsed_hours = round(elapsed_seconds / 3600, 2)

    # Signal GPU worker to stop and wait for it
    event_stop_gpuworker.set()
    process_gpu_worker.join()

    # Stop CPU profiling
    cpu_profiler.disable()
    with open(str(config_base['path_cpulogfile']), "w") as f:
        stats_stream = pstats.Stats(cpu_profiler, stream=f).sort_stats('time')
        stats_stream.print_stats()

    # Stop GPU profiling
    event_stop_gpuprofiler.set()
    thread_gpuprofiler.join()

    # End of main logger
    logger.info(f"[MAIN] Completed simulation. {elapsed_hours} hours elapsed from {start_time::%Y-%m-%d %H:%M:%S} to {end_time::%Y-%m-%d %H:%M:%S}.")


if __name__ == '__main__':
    main()
