import csv
import os
import re
import subprocess
from subprocess import call
import yaml
import time
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
import sys
import glob
import shutil
from IPython.display import clear_output
from pathlib import Path
import logging
import gc
import threading
import cProfile
import pstats
import uuid
import warnings
import argparse

import petsc4py
petsc4py.init()
from petsc4py import PETSc
from numba import cuda, float32, int32
import math
import cupy as cp
from cupy.fft import fftfreq, fft2, fftshift
import multiprocessing as mp
from scipy.fft import fftn, dctn, next_fast_len
import numpy as np
import heapq
from collections import Counter

# Usage: python /home1/binhaoli/flow_simulation/test11_op_pressureField/run.py --path_workdir /home1/binhaoli/flow_simulation/test11_op_pressureField --num_mc 10 --field_size 100 --hurst 0.25 --minpadding 0 --std 6

# Add libgfortran.so.5 and libstdc++.so.6 in /apps/generic/gcc/13.3.0/lib64 as an environment variable
os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
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

def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_workdir', required=True, type=str, help="Path to the working directory where flow packages lie in: str")
    parser.add_argument('--num_mc', required=True, nargs='+', type=int, help='Number of MC: int list')
    parser.add_argument('--field_size', required=True, nargs='+', type=int, help='Field size: int list')
    parser.add_argument('--hurst', required=True, type=float, help="Hurst exponent")
    parser.add_argument('--minpadding', type=int, default=0, help="Minimal padding for fBm generation")
    parser.add_argument('--c', type=float, default=None, help="Coefficient for semivariogram for fBm generation")
    parser.add_argument('--std', type=float, default=None, help="Standard deviation for fBm generation")
    args = parser.parse_args()
    print("\nAll input parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

    print("CUDA available:", cuda.is_available())
        
    mp.set_start_method('spawn')
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    path_workdir = Path(args.path_workdir).expanduser().resolve()

    # Base configuration dictionary
    config_base = {
        'num_mc': args.num_mc,
        'dx': 1, 
        'dy': 1, 
        'dz': 1,
        'field_size': args.field_size,
        'fbm_arg': {
            'hurst': args.hurst,
            'std': args.std,
            'c': args.c,
            'minpadding': args.minpadding,
        },
        'path_simulation': path_workdir,
        'is_save_lenpath_mc': True,
        'headSolver_arg': {
            'dh': 0.01, 
            'rtol': 1e-10, 
            'atol': 1e-20, 
            'max_it': 10000
        }
    }

    config_base['particleTracking_arg'] = {
        'density_particle': 20, 
        'eps_speed': 1e-30, 
        'ds_min': 1e-20, 
        'ds_max': 0.5*min(config_base['dx'], config_base['dy']), 
        'tol': 1e-4, 
        'y_policy': "stop", 
        'max_points': 100000, 
        'max_iters': 1000000, 
        'threads_per_block': 256,
    }
    config_base['path_log'] = config_base['path_simulation'] / 'log'
    config_base['path_mainlogfile'] = config_base['path_log'] / f'simulation_{timestamp}.log'
    config_base['path_gpulogfile'] = config_base['path_log'] / f'gpu_profile_{timestamp}.log'
    config_base['path_cpulogfile'] = config_base['path_log'] / f'cpu_profile_{timestamp}.log'
    
    # Ensure output directory exists
    config_base['path_simulation'].mkdir(parents=True, exist_ok=True)
    config_base['path_log'].mkdir(parents=True, exist_ok=True)

    # Setup main logger
    logger = get_logger(path_logfile = str(config_base['path_mainlogfile']))
    logger.info(f"[MAIN] Starting simulation.")
    
    # Setup GPU profiling
    event_stop_gpuprofiler = threading.Event()
    thread_gpuprofiler = threading.Thread(target=log_gpu_usage, kwargs={
        'path_logfile': str(config_base['path_gpulogfile']),
        'interval': 1.0,
        'event_stop': event_stop_gpuprofiler
    })
    thread_gpuprofiler.start()

    # Setup CPU profiling
    cpu_profiler = cProfile.Profile()
    cpu_profiler.enable()

    # Setup GPU Worker process
    manager = mp.Manager()
    result_dict = manager.dict()
    queue_gpu = mp.Queue()
    event_stop_gpuworker = mp.Event()  # Separate event for the GPU worker process
    process_gpu_worker = mp.Process(target=gpu_worker, args=(config_base, event_stop_gpuworker, queue_gpu, result_dict), name="GPU-Worker")
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

def run_field_size_simulation(idx_size, config_base, queue_gpu, result_dict):
    """Run simulations for a given field size with parallelized Monte Carlo runs."""

    # Update config for this field size
    field_size_ = config_base['field_size'][idx_size]
    logger = get_logger(path_logfile = str(config_base['path_mainlogfile']))
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
    config['path_size_lenpath'] = config['path_size'] / 'len_path'
    config['path_size_path'] = config['path_size'] / 'path'
    config['path_size_logk'] = config['path_size'] / 'logk_field'

    if config['path_size'].exists() and config['path_size'].is_dir():
        shutil.rmtree(config['path_size'])
    config['path_size'].mkdir(parents=True, exist_ok=True)
    config['path_size_lenpath'].mkdir(parents=True, exist_ok=True)
    config['path_size_path'].mkdir(parents=True, exist_ok=True)
    config['path_size_logk'].mkdir(parents=True, exist_ok=True)

    # Initialize a fBm generator
    if config['fbm_arg']['std'] is not None:
        c = compute_c(field_size_, config['fbm_arg']['hurst'], config['fbm_arg']['std'])
    elif config['fbm_arg']['c'] is not None:
        c = config['fbm_arg']['c']
        
    fbm_gen = fBmGenerator(
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
    len_lrp_sm = np.array([r[0] for r in mc_result])
    len_fp_sm = np.array([r[1] for r in mc_result])
    np.save(str(config['path_size_lenpath'] / f'len_lrp_size{field_size_}.npy'), len_lrp_sm)
    np.save(str(config['path_size_lenpath'] / f'len_fp_size{field_size_}.npy'), len_fp_sm)
    logger.info(f"[SIZE {field_size_}] Paths saved.")
    logger.info(f"[SIZE {field_size_}] Completed simulation.")
    
    return idx_size, len_lrp_sm, len_fp_sm
    
def run_unit_simulation(config, num_mc_, num_trial=20):
    """
    Run a single Monte Carlo simulation with controlled retry logic.
    """
    logger = get_logger(path_logfile=str(config['path_mainlogfile']))
    logger.info(f"[MC {num_mc_}] Starting simulation (up to {num_trial} trials).")

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
        lrp, total_R, s_node, e_node, Rmap = dijkstra_2d_least_resistance(k_field, source, target, average='harmonic') 
        len_lrp = euclidean_length(lrp)

        # Check LRP geometry
        if not is_path_valid(lrp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid LRP, retrying next trial.")
            continue

        # === 3. Compute FP ===
        logger.info(f"[MC {num_mc_}] Computing FP.")
        h, msg, is_successful = solve_head(k_field, config_mc['x'], config_mc['y'], 
                                            dh=config_mc['headSolver_arg']['dh'], 
                                            rtol=config_mc['headSolver_arg']['rtol'], 
                                            atol=config_mc['headSolver_arg']['atol'], 
                                            max_it=config_mc['headSolver_arg']['max_it'])
        if not is_successful:
            logger.warning(f"[MC {num_mc_}] Head not converged, retrying next trial. Error: {msg}")
            continue
        # Velocity field    
        vx, vy = compute_velocity(k_field, config_mc['x'], config_mc['y'], h)

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
        px, py, pt, L, st, info_code, info_value, info_speed = gpu_call('particle_tracking_rk45', args_particleTracking, f'[MC {num_mc_}] Particle Tracking')
        # 0=success, 1=stagnation, 4=other errors
        count = Counter(st.tolist())
        if count[0]/nP < 0.5 or (count[1] + count[4])/nP > 0.1:
            logger.warning(f"[MC {num_mc_}] Invalid velocity field, retrying next trial. Error: {count}")
            continue
            
        # Fastest path
        res = fastest_trajectory(px, py, pt, L, st)
        fp = np.vstack((res["x"], res["y"])).T
        fp = fp[fp[:,0] <= config_mc['x_cp']] # Trim the traj
        len_fp = euclidean_length(fp)

        # Check FP geometry
        if not is_path_valid(fp, domain_box, len_min=np.abs(x1 - x0)):
            logger.warning(f"[MC {num_mc_}] Invalid FP, retrying next trial.")
            continue

        # === 4. Save and exit successfully ===
        np.save(config_mc['path_size_lenpath'] / f"len_lrp_size{config_mc['field_size_']}_mc{num_mc_}.npy", len_lrp)
        np.save(config_mc['path_size_lenpath'] / f"len_fp_size{config_mc['field_size_']}_mc{num_mc_}.npy", len_fp)
        np.save(config_mc['path_size_path'] / f"lrp_size{config_mc['field_size_']}_mc{num_mc_}.npy", lrp)
        np.save(config_mc['path_size_path'] / f"fp_size{config_mc['field_size_']}_mc{num_mc_}.npy", fp)
        logger.info(f"[MC {num_mc_}] Paths saved.")

        logger.info(f"[MC {num_mc_}] Trial {trial + 1} successful.")
        break

    else:
        len_lrp = np.nan
        len_fp = np.nan
        logger.error(f"[MC {num_mc_}] All {num_trial} trials failed to produce valid output.")

    return len_lrp, len_fp
    
def generate_rectangular_array(vertex_pair):
    '''Generate a rectangular grid of points in a given dimension.'''

    # Ensure p0 and p1 are NumPy arrays
    p0 = np.array(vertex_pair[0], dtype=int)
    p1 = np.array(vertex_pair[1], dtype=int)
    
    # Determine the number of dimensions
    dim = p0.shape[0]
    
    # Create the grid points for each dimension
    grid_points = []
    
    for i in range(dim):
        grid_points.append(np.arange(p0[i], p1[i] + 1))
    
    # Create the meshgrid for the given dimension
    grid = np.meshgrid(*grid_points, indexing='ij')
    
    # Stack the grids into a single array (this will be the final grid)
    grid_stacked = np.stack(grid, axis=-1)
    
    return grid_stacked
                    
def euclidean_length(trajectory):
    """
    Calculate the Euclidean length of a trajectory using CuPy for GPU acceleration.
    Args:
        trajectory (np.ndarray): D-dimensional array of shape (N, D) representing the trajectory points.
    Returns:
        float: Euclidean length of the trajectory.
    """
    
    trajectory = cp.asarray(trajectory)
    
    diff = trajectory[1:] - trajectory[:-1]  # Differences across all dimensions
    length = cp.sum(cp.sqrt(cp.sum(diff**2, axis=-1)))
    length_np = cp.asnumpy(length)

    del trajectory, diff, length
    gc.collect()
    
    return length_np

def get_logger(name='simulation', path_logfile='simulation.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s')
        file_handler = logging.FileHandler(path_logfile, mode='a')
        console_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        lines = result.stdout.strip().split('\n')
        usage_data = []
        for line in lines:
            used, free = [item.strip() for item in line.split(',')]
            usage_data.append(f"Memory: {int(used):,}MB used, {int(free):,}MB free.")
        return ' || '.join(usage_data)
    except Exception as e:
        return f"GPU info error: {e}"

def generate_logk_field(config):
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

def is_path_valid(path, box, len_min):
    """
    Check if a path stays within a bounding box and exceeds a minimum length.
    box = (x0, x1, y0, y1)
    """
    x0, x1, y0, y1 = box
    x, y = path[:, 0], path[:, 1]
    # print(f'minx = {min(x)}, maxx = {max(x)}, miny = {min(y)}, maxy = {max(y)}')
    # print(f'x0 = {x0}, x1 = {x1}, y0 = {y0}, y1 = {y1}')
    in_box = (min(x) >= x0) & (max(x) <= x1) & (min(y) >= y0) & (max(y) <= y1)
    # in_box = (min(y) >= y0) & (max(y) <= y1)
    long_enough = euclidean_length(path) >= len_min
    # print(in_box, long_enough)
    return in_box and long_enough

def log_gpu_usage(path_logfile='gpu_profile.log', interval=1.0, event_stop=None):
    """Periodically logs GPU usage to a file using nvidia-smi."""
    with open(path_logfile, 'w') as f:
        f.write("timestamp, gpu_utilization [%], memory_used [MB], memory_free [MB]\n")
        while not event_stop.is_set():
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=timestamp,utilization.gpu,memory.used,memory.free',
                        '--format=csv,noheader,nounits'
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                f.write(result.stdout.strip() + '\n')
                time.sleep(interval)
            except Exception as e:
                f.write(f"# Error: {e}\n")
                break

def gpu_worker(config, event_stop_gpuworker, queue_gpu, result_dict):
    """
    GPU worker process: pulls tasks from the GPU task queue, processes them,
    and writes results into the shared result dictionary.
    """
    logger = get_logger(path_logfile=str(config['path_mainlogfile']))
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
            result = run_par2(*args, **kwargs)
        elif func_name == 'particle_tracking_rk45':
            result = particle_tracking_rk45(*args, **kwargs)
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

class GaussianRandomFieldCircEmbed:
    """
    Gaussian random field via circulant embedding.
    """
    def __init__(self, mean, cov, pts,
                 minpadding=0):
        self.mean = mean
        self.cov  = cov
        self.pts  = pts
        
        normedpts = [p - p[0] for p in pts]
        
        # compute minimal circulant‐embedding size in each dim
        pad = minpadding
        if np.isscalar(pad):
            pads = [pad]*len(pts) # if minpadding is scalar, use same in each dim
        else:
            pads = pad
        dims = [
            self.circulant_minsize(cov, normedpts[i], pads[i])
            for i in range(len(pts))
        ]
        
        # compute the embedded covariance eigenvalues
        Λ = self.circulant_eigvals(cov, normedpts, dims)
        
        # zero out any small negative λ due to numerical error,
        #    track how many and how negative, then scale by sqrt(λ/M)
        M = np.prod(dims)
        neg_mask = Λ < 0
        n_neg = int(neg_mask.sum())
        λ_min = None
        if n_neg:
            λ_min = float(Λ[neg_mask].min())
            Λ[neg_mask] = 0.0
        pos_mask = ~neg_mask
        Λ[pos_mask] = np.sqrt(Λ[pos_mask] / M)
        
        if n_neg:
            warnings.warn(
                f"{n_neg} negative eigenvalues ≥ {λ_min:.3e} were set to zero; "
                "consider increasing padding."
            )
        
        # store data:  v = sqrt(eigs/M)  and  shape_emb = tuple(dims)
        self.data = (Λ, tuple(dims))
        
    def sample(self):
        """
        Single realization.  xi must be shape == self.data[1].
        """
        Λ, shape_emb = self.data
        xi = np.random.randn(*shape_emb)
        # 1) spectral weights
        w = Λ * xi
        # 2) forward FFT over all dims
        w = fftn(w)
        # 3) crop back to original grid and extract real+imag
        orig_shape = tuple(len(p) for p in self.pts)
        slices = tuple(slice(0, n) for n in orig_shape)
        w_crop = w[slices]
        return self.mean + (w_crop.real + w_crop.imag)

    @staticmethod
    def circulant_minsize(cov, pts, minpadding):
        n = len(pts)
        base = n + minpadding - (1 if cov.is_even else 0)
        return 2 * next_fast_len(base) # SciPy’s next_fast_len returns the smallest ≥ base with 2,3,5 factors

    @staticmethod
    def circulant_eigvals(cov, pts, dims):
        pts = [np.asarray(p) for p in pts]
        starts = [p[0] for p in pts]
        steps = [p[1] - p[0] for p in pts]
        N = len(dims)
        is_even = getattr(cov, 'is_even', False)
        
        # Vectorized covariance evaluation
        cov_vec = np.vectorize(cov, signature='(n)->()')
        
        if is_even:
            half_shape = tuple(d//2 + 1 for d in dims)
            coords = [starts[d] + np.arange(half_shape[d]) * steps[d] for d in range(N)]
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            C = cov_vec(mesh)
            C = dctn(C, type=1, norm=None)
            idx = [np.minimum(np.arange(d), d - np.arange(d)) for d in dims]
            mesh_idx = np.ix_(*idx)
            return C[mesh_idx]
        else:
            C = np.zeros(dims, dtype=float)
            dims2 = [d + 2 for d in dims]
            mids  = [d2 // 2 for d2 in dims2]
            grid = [np.arange(d) for d in dims]
            # Build signed offsets
            coords = []
            for d in range(N):
                idx = grid[d]
                pos = np.where(idx < mids[d], idx, idx - dims[d])
                coords.append(starts[d] + pos * steps[d])
            mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
            mask = np.ones(dims, dtype=bool)
            for d in range(N):
                mask &= ((grid[d] + 1) != mids[d]).reshape(
                    *(dims[j] if j == d else 1 for j in range(N))
                )
            C[mask] = cov_vec(mesh[mask])
            return np.real(np.fft.fftn(C))

class PrefbmCov:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_even = True
        self.c0 = None
        self.c2 = None
    def __call__(self, x):
        r = np.linalg.norm(x)
        α = self.alpha
        if 0 < α <= 1.5:
            self.c0 = 1 - α/2
            self.c2 = α/2
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0- r**α + self.c2*r**2
            else:
                return 0.0
        elif 1.5 < α < 2:
            self.c0 = 1 - α/6 - α**2/6
            self.c2 = α*(5+2*α)/18
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**α + self.c2*r**2
            elif r <= 2:
                return α*(2-α)/18 * (2-r)**3 / r
            else:
                return 0.0
        else:
            raise ValueError("α must lie in (0, 2)")

class fBmGenerator:
    """
    Generate one realization of a d‐dimensional fBm field on pts via:
      1) circulant‐embedded GRF (mean zero, PrefbmCov)
      2) additive linear trend ∑ x_i·X_i
      3) rescaling
    
    Usage:
      fbm = fBmGenerator(c=2.0, alpha=0.5, pts=(x_grid, y_grid), minpadding=128)
      field = fbm.sample()
    """
    def __init__(self, *, c, alpha, num_pt, dim=2, minpadding=0):
        self.c          = c
        self.alpha      = alpha
        self.num_pt     = num_pt
        self.pts        = (np.arange(num_pt) / num_pt,) * dim
        self.minpadding = minpadding if alpha <= 1.5 else minpadding+num_pt
        self.cov_c0 = None
        self.cov_c2 = None

        # 1) set up zero‐mean GRF with your Cov class
        mean = np.zeros(tuple(len(p) for p in self.pts))
        cov  = PrefbmCov(alpha)
        self._prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, self.pts, minpadding=self.minpadding)
        self.cov_c0 = self._prefbm_gen.cov.c0
        self.cov_c2 = self._prefbm_gen.cov.c2
        self._σ_trend = np.sqrt(2*self.cov_c2)

    def sample(self):
        prefbm = self._prefbm_gen.sample()

        # build the linear trend ∑ x_i·X_i
        coords = np.meshgrid(*self.pts, indexing="ij")
        Xs     = np.random.normal(scale=self._σ_trend, size=len(coords))
        trend  = sum(coord * X for coord, X in zip(coords, Xs))

        # Add trend and rescale
        fbm_c0_unit = prefbm + trend # Var(Z(x)) = c0 + 2*c2*|x|^2
        fbm = np.sqrt(self.c) * self.num_pt ** (self.alpha/2) * (fbm_c0_unit - fbm_c0_unit[0,0])
        
        return fbm

def compute_c(l, H, std_target):
    """
    Compute scale factor c needed so that a 2D fBm field of size l×l
    with Var[X] = 2*c*r^(2H) has marginal std = std_target.
    """
    coords = np.arange(l) - (l - 1) / 2
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    r = np.sqrt(X**2 + Y**2)

    # compute average r^(2H)
    avg_r2H = np.mean(r**(2 * H))
    
    # solve std_target^2 = 2*c*avg_r2H  =>  c = std_target^2 / (2*avg_r2H)
    return std_target**2 / (2 * avg_r2H)

def solve_head(K, x, y, dh=0.01, rtol=1e-6, atol=1e-20, max_it=10000):
    """
    Solve ∇·(K∇h) = 0 in a 2D domain with:
    - Dirichlet BCs on left/right: h(left)=dh*x_len, h(right)=0
    - No-flow BCs on top/bottom.
    
    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx).
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions.
    dh : float
        Mean hydraulic head gradient between left and right boundaries.
    
    Returns
    -------
    h : 2D np.ndarray
        Computed hydraulic head field.
    msg : str
        PETSc convergence message.
    """
    ny, nx = K.shape
    dx = np.diff(np.concatenate(([0], x + 0.5*(x[1]-x[0]))))
    dy = np.diff(np.concatenate(([0], y + 0.5*(y[1]-y[0]))))
    dz = 1.0

    def lid(i, j): return i*nx + j
    def hmean(a, b): return 2*a*b/(a+b + 1e-300)

    dx_face = np.empty(nx+1)
    dx_face[1:-1] = 0.5*(dx[:-1] + dx[1:])
    dx_face[0] = dx[0]
    dx_face[-1] = dx[-1]

    dy_face = np.empty(ny+1)
    dy_face[1:-1] = 0.5*(dy[:-1] + dy[1:])
    dy_face[0] = dy[0]
    dy_face[-1] = dy[-1]

    # PETSc assembly
    N = nx * ny
    A = PETSc.Mat().create()
    A.setSizes((N, N))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setPreallocationNNZ(5)

    b = PETSc.Vec().createMPI(N)
    b.set(0.0)

    LEFT_VAL, RIGHT_VAL = dh*np.abs(x[-1]-x[0]), 0.0

    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = divmod(row, nx)
        is_left = (j == 0)
        is_right = (j == nx-1)

        if is_left or is_right:
            A.setValue(row, row, 1.0)
            b.setValue(row, LEFT_VAL if is_left else RIGHT_VAL)
            continue

        Ax = dy[i]*dz
        Ay = dx[j]*dz

        KW = hmean(K[i, j], K[i, j-1])
        GW = KW * Ax / dx_face[j]

        KE = hmean(K[i, j], K[i, j+1])
        GE = KE * Ax / dx_face[j+1]

        GS = hmean(K[i, j], K[i-1, j]) * Ay / dy_face[i] if i > 0 else 0.0
        GN = hmean(K[i, j], K[i+1, j]) * Ay / dy_face[i+1] if i < ny-1 else 0.0

        diag = GE + GW + GN + GS
        A.setValue(row, row, diag)

        if j-1 == 0:
            b.setValue(row, GW*LEFT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j-1), -GW)

        if j+1 == nx-1:
            b.setValue(row, GE*RIGHT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j+1), -GE)

        if i > 0:
            A.setValue(row, lid(i-1, j), -GS)
        if i < ny-1:
            A.setValue(row, lid(i+1, j), -GN)

    A.assemblyBegin(); A.assemblyEnd()
    b.assemblyBegin(); b.assemblyEnd()

    x_vec = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    try:
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
    except PETSc.Error:
        pc.setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    ksp.setFromOptions()
    ksp.solve(b, x_vec)

    h_flat = x_vec.getArray()
    h = h_flat.reshape((ny, nx))

    msg = (f"ConvergedReason: {ksp.getConvergedReason()}, "
           f"Iterations: {ksp.getIterationNumber()}, "
           f"Residual: {ksp.getResidualNorm():.3e}")
    is_success = True if (ksp.getConvergedReason() > 0) else False
    return h, msg, is_success

def compute_velocity(K, x, y, h):
    """
    Compute Darcy velocity components (vx, vy) from head field h(x,y).

    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx)
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions
    h : 2D np.ndarray
        Hydraulic head field, shape (ny, nx)

    Returns
    -------
    vx, vy : 2D np.ndarray
        Darcy velocity components in x and y directions
        vx = -K * ∂h/∂x, vy = -K * ∂h/∂y
    """
    ny, nx = h.shape
    dhdx = np.zeros_like(h)
    dhdy = np.zeros_like(h)

    # --- x-derivative (central interior, one-sided edges)
    for i in range(ny):
        dhdx[i, 1:-1] = (h[i, 2:] - h[i, :-2]) / (x[2:] - x[:-2])
        dhdx[i, 0] = (h[i, 1] - h[i, 0]) / (x[1] - x[0])
        dhdx[i, -1] = (h[i, -1] - h[i, -2]) / (x[-1] - x[-2])

    # --- y-derivative (central interior, one-sided edges)
    for j in range(nx):
        dhdy[1:-1, j] = (h[2:, j] - h[:-2, j]) / (y[2:] - y[:-2])
        dhdy[0, j] = (h[1, j] - h[0, j]) / (y[1] - y[0])
        dhdy[-1, j] = (h[-1, j] - h[-2, j]) / (y[-1] - y[-2])

    # --- Darcy velocity
    vx = -K * dhdx
    vy = -K * dhdy
    return vx, vy

# BEGIN: RK45 parellel particle tracking 
@cuda.jit(device=True, inline=True)
def find_le_index(arr, n, val):
    """
    Return largest idx i such that arr[i] <= val < arr[i+1]; clamp to [0, n-2].
    Assumes arr is strictly increasing. Binary search for speed would be nicer,
    but linear probe is fine within a few iters if grids are modest; switch to
    binary if n is large.
    """
    # binary search
    lo = 0
    hi = n - 1
    if val <= arr[0]:
        return 0
    if val >= arr[n-2]:
        return n-2
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo

@cuda.jit(device=True, inline=True)
def bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y, out_v):
    """
    out_v[0]=vx(x,y), out_v[1]=vy(x,y), bilinear on rect grid (x fastest).
    Arrays vx,vy have shape (ny, nx) in row-major; we index [iy, ix].
    """
    # Clamp to interior cell
    if x <= xgrid[0]:
        ix = 0
    elif x >= xgrid[nx-1]:
        ix = nx - 2
    else:
        ix = find_le_index(xgrid, nx, x)

    if y <= ygrid[0]:
        iy = 0
    elif y >= ygrid[ny-1]:
        iy = ny - 2
    else:
        iy = find_le_index(ygrid, ny, y)

    x0 = xgrid[ix]
    x1 = xgrid[ix+1]
    y0 = ygrid[iy]
    y1 = ygrid[iy+1]
    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)

    # Corners
    i00 = iy * nx + ix
    i10 = iy * nx + (ix+1)
    i01 = (iy+1) * nx + ix
    i11 = (iy+1) * nx + (ix+1)

    vx00 = vx[i00]; vx10 = vx[i10]; vx01 = vx[i01]; vx11 = vx[i11]
    vy00 = vy[i00]; vy10 = vy[i10]; vy01 = vy[i01]; vy11 = vy[i11]

    vx0 = vx00 + tx * (vx10 - vx00)
    vx1 = vx01 + tx * (vx11 - vx01)
    vy0 = vy00 + tx * (vy10 - vy00)
    vy1 = vy01 + tx * (vy11 - vy01)
    out_v[0] = vx0 + ty * (vx1 - vx0)
    out_v[1] = vy0 + ty * (vy1 - vy0)

@cuda.jit(device=True, inline=True)
def norm2(x, y):
    return math.sqrt(x*x + y*y)

@cuda.jit(device=True)
def rhs_arclength(x, y, xgrid, ygrid, vx, vy, nx, ny, eps_speed, out_rhs, out_speed):
    """
    For z=[x,y,t], dz/ds = [vx/|v|, vy/|v|, 1/|v|].
    """
    v = cuda.local.array(2, dtype=float32)
    bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y, v)
    sp = norm2(v[0], v[1])
    if not math.isfinite(sp) or sp < eps_speed:
        out_rhs[0] = 0.0
        out_rhs[1] = 0.0
        out_rhs[2] = 1e30  # not used; event will stop
        out_speed[0] = sp
        return
    inv = 1.0 / sp
    out_rhs[0] = v[0] * inv
    out_rhs[1] = v[1] * inv
    out_rhs[2] = inv
    out_speed[0] = sp

@cuda.jit
def rk45_arclength_kernel(xgrid, ygrid,
                          vx, vy, nx, ny,
                          seeds_x, seeds_y,
                          right_x, xmin, xmax, ymin, ymax,
                          eps_speed, ds_min, ds_max, tol,
                          y_policy,
                          max_points, max_iters,
                          A_tbl, B5_tbl, B4_tbl,
                          paths_x, paths_y, paths_t, lengths, status,
                          info_code, info_value, info_speed):
    """
    One thread per particle. Writes polyline to paths_*[pid, :]
    status: 0=ok/hit_right, 1=stagnation, 2=left_domain_x, 3=left_domain_y, 4=fail
    """
    pid = cuda.grid(1)
    nP = seeds_x.size
    if pid >= nP:
        return

    x = float(seeds_x[pid])
    y = float(seeds_y[pid])
    t = 0.0

    # init
    count = 0
    # write initial point
    if count < max_points:
        paths_x[pid, count] = x
        paths_y[pid, count] = y
        paths_t[pid, count] = t
        count += 1

    ds = ds_max * 0.25  # initial guess
    iters = 0
    st = 4  # default fail until exit well

    # local buffers
    kx = cuda.local.array(7, dtype=float32)
    ky = cuda.local.array(7, dtype=float32)
    kt = cuda.local.array(7, dtype=float32)
    rhs = cuda.local.array(3, dtype=float32)
    spv = cuda.local.array(1, dtype=float32)

    while iters < max_iters and count < max_points:
        iters += 1

        # Event: hit right boundary
        if x >= right_x:
            st = 0
            break
        # Event: domain in x
        if x <= xmin or x >= xmax:
            st = 2
            break
        # y policy pre-check
        if y < ymin or y > ymax:
            if y_policy == 0:  # stop
                st = 3
                break
            elif y_policy == 1:  # reflect
                # reflect inside [ymin, ymax] with folding
                Y = y
                span = (ymax - ymin)
                # modulo reflection
                Y = ymin + math.fmod(Y - ymin, 2.0*span)
                if Y < ymin: Y += 2.0*span
                if Y > ymax: Y = ymax - (Y - ymax)
                y = Y
            elif y_policy == 2:  # clip
                if y < ymin: y = ymin
                if y > ymax: y = ymax

        # Guard ds
        if ds < ds_min: ds = ds_min
        if ds > ds_max: ds = ds_max

        # RK stages (Dormand–Prince: 6 stages for 5th order solution)
        # k1
        rhs_arclength(x, y, xgrid, ygrid, vx, vy, nx, ny, eps_speed, rhs, spv)
        sp = spv[0]
        if not math.isfinite(sp) or sp < eps_speed:
            st = 1  # stagnation
            break
        kx[0] = rhs[0]; ky[0] = rhs[1]; kt[0] = rhs[2]

        # k2..k6
        for s in range(1, 6):
            xi = x; yi = y; ti = t
            for j in range(s):
                a = A_tbl[s, j]
                xi += ds * a * kx[j]
                yi += ds * a * ky[j]
                ti += ds * a * kt[j]
            rhs_arclength(xi, yi, xgrid, ygrid, vx, vy, nx, ny, eps_speed, rhs, spv)
            kx[s] = rhs[0]; ky[s] = rhs[1]; kt[s] = rhs[2]

        # 5th-order proposal
        x5 = x; y5 = y; t5 = t
        for j in range(6):
            x5 += ds * B5_tbl[j] * kx[j]
            y5 += ds * B5_tbl[j] * ky[j]
            t5 += ds * B5_tbl[j] * kt[j]

        # 4th-order estimate (note: B4 has 7 entries; use the first 6 with k0..k5, then add the last with k0 again as before)
        x4 = x; y4 = y; t4 = t
        for j in range(6):
            x4 += ds * B4_tbl[j] * kx[j]
            y4 += ds * B4_tbl[j] * ky[j]
            t4 += ds * B4_tbl[j] * kt[j]
        x4 += ds * B4_tbl[6] * kx[0]
        y4 += ds * B4_tbl[6] * ky[0]
        t4 += ds * B4_tbl[6] * kt[0]

        # Error measure on position (scaled)
        ex = x5 - x4
        ey = y5 - y4
        err = math.sqrt(ex*ex + ey*ey) + 1e-30

        if err <= tol:
            # accept step
            x = x5; y = y5; t = t5
            if count < max_points:
                paths_x[pid, count] = x
                paths_y[pid, count] = y
                paths_t[pid, count] = t
                count += 1
            # adapt ds (PI-controller-ish)
            ds = ds * min(5.0, max(0.2, 0.9 * math.pow(tol / err, 0.2)))
        else:
            # reject & shrink step
            ds = ds * max(0.1, 0.5 * math.pow(tol / err, 0.25))

        # Detect crossing of right_x between (x,y) and (x5,y5)
        if (x < right_x) and (x5 >= right_x):
            # linear interpolation fraction in this step
            denom = (x5 - x) + 1e-30
            tau = (right_x - x) / denom
            # clamp tau to [0,1]
            if tau < 0.0: tau = 0.0
            if tau > 1.0: tau = 1.0
            x = right_x
            y = y + tau * (y5 - y)
            t = t + tau * (t5 - t)
            if count < max_points:
                paths_x[pid, count] = x
                paths_y[pid, count] = y
                paths_t[pid, count] = t
                count += 1
            st = 0  # success at the wall
            break
       
    # Default info values
    info_c = -1
    info_v = 0.0
    info_s = sp
    
    # Primary outcome already determined as st = 0..3
    # Now record secondary diagnostics
    if ds < ds_min:
        info_c, info_v = 4, ds
    elif ds > ds_max:
        info_c, info_v = 5, ds
    elif err > tol:
        info_c, info_v = 6, err
    elif count >= max_points:
        info_c, info_v = 7, float(count)
    elif iters >= max_iters:
        info_c, info_v = 8, float(iters)
    elif not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(t):
        info_c, info_v = 9, 0.0
    
    info_code[pid]  = info_c
    info_value[pid] = info_v
    info_speed[pid] = info_s
    
    lengths[pid] = count
    status[pid]  = st if (st != 4 or iters < max_iters) else 4  # 4=fail/iters/exhaustion

def particle_tracking_rk45(params):
    """
    Parameters
    ----------
    xgrid, ygrid : 1D arrays defining rect grid (strictly increasing)
    vx, vy       : 2D arrays shape (ny, nx), velocity at grid nodes
    seeds_x/y    : particle seeds (1D, same length)
    right_x      : stopping x (default = xgrid.max())
    eps_speed    : stagnation threshold on |v|
    ds_min/max   : min/max arclength step
    tol          : RK45 error tolerance on position
    y_policy     : boundary policy in y
    max_points   : preallocated points per particle
    max_iters    : safety cap on RK steps
    dtype        : float32 recommended for speed
    """
    
    # Set required parameters
    x = params["x"]
    y = params["y"]
    vx = params["vx"]
    vy = params["vy"]
    seeds_x = params["seeds_x"]
    seeds_y = params["seeds_y"]

    # Set optional parameters with defaults
    right_x = params.get("right_x", None)
    eps_speed = params.get("eps_speed", 1e-16)
    ds_min = params.get("ds_min", 1e-5)
    ds_max = params.get("ds_max", 1.0)
    tol = params.get("tol", 1e-4)
    y_policy = params.get("y_policy", "stop") # "stop" | "reflect" | "clip"
    max_points = params.get("max_points", 2000)
    max_iters = params.get("max_iters", 200000)
    threads_per_block = params.get("threads_per_block", 256)
    stream = params.get("stream", None)
    dtype = params.get("dtype", np.float32)
    
    # Dormand–Prince RK45 (embedded 4/5)
    # Coefficients (float32)
    # Dormand–Prince RK45 coefficients (padded for rectangular shape)
    A = np.zeros((6, 6), dtype=np.float32)
    A[1,0] = 1/5
    A[2,0:2] = [3/40, 9/40]
    A[3,0:3] = [44/45, -56/15, 32/9]
    A[4,0:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
    A[5,0:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    
    C = np.array([0., 1/5, 3/10, 4/5, 8/9, 1.], dtype=np.float32)
    
    B5 = np.array([35/384, 0., 500/1113, 125/192, -2187/6784, 11/84], dtype=np.float32)
    B4 = np.array([5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float32)
    
    # Upload to device
    A_d = cuda.to_device(A)
    C_d = cuda.to_device(C)
    B5_d = cuda.to_device(B5)
    B4_d = cuda.to_device(B4)

    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)
    vx    = np.asarray(vx,    dtype=dtype, order='C')
    vy    = np.asarray(vy,    dtype=dtype, order='C')
    seeds_x = np.asarray(seeds_x, dtype=dtype)
    seeds_y = np.asarray(seeds_y, dtype=dtype)

    ny, nx = vx.shape
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if right_x is None:
        right_x = xmax

    # Transfer to device
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_vx    = cuda.to_device(vx.ravel())  # flatten to 1D for simple indexing
    d_vy    = cuda.to_device(vy.ravel())
    d_seeds_x = cuda.to_device(seeds_x)
    d_seeds_y = cuda.to_device(seeds_y)

    nP = seeds_x.size
    paths_x = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    paths_y = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    paths_t = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    lengths = cuda.device_array(nP, dtype=np.int32, stream=stream)
    status  = cuda.device_array(nP, dtype=np.int32, stream=stream)

    # --- Additional diagnostic arrays ---
    info_code  = cuda.device_array(nP, dtype=np.int32, stream=stream)
    info_value = cuda.device_array(nP, dtype=dtype, stream=stream)
    info_speed = cuda.device_array(nP, dtype=dtype, stream=stream)

    ypol = {"stop":0, "reflect":1, "clip":2}[y_policy]

    blocks = (nP + threads_per_block - 1) // threads_per_block

    rk45_arclength_kernel[blocks, threads_per_block, stream](
        d_x, d_y,
        d_vx, d_vy, np.int32(nx), np.int32(ny),
        d_seeds_x, d_seeds_y,
        dtype(right_x), dtype(xmin), dtype(xmax), dtype(ymin), dtype(ymax),
        dtype(eps_speed), dtype(ds_min), dtype(ds_max), dtype(tol),
        np.int32(ypol),
        np.int32(max_points), np.int32(max_iters),
        # NEW: pass coeff arrays here
        A_d, B5_d, B4_d,
        paths_x, paths_y, paths_t, lengths, status,
        info_code, info_value, info_speed
    )
    cuda.synchronize()
    info_code_host  = info_code.copy_to_host()
    info_value_host = info_value.copy_to_host()
    info_speed_host = info_speed.copy_to_host()

    return (paths_x.copy_to_host(), paths_y.copy_to_host(), paths_t.copy_to_host(),
            lengths.copy_to_host(), status.copy_to_host(),
            info_code_host, info_value_host, info_speed_host)
# END: RK45 parellel particle tracking

def dijkstra_2d_least_resistance(K, start_region, end_region, average='harmonic'):
    """
    Finds the least-resistance path on a 2D conductivity lattice using Dijkstra's algorithm,
    considering 8-neighbor connectivity and user-selected averaging (harmonic or arithmetic)
    of bond conductivities.

    Args:
        K (np.ndarray): 2D array of conductivities (higher = easier to move).
        start_region (list[tuple[int,int]]): List of (row, col) coordinates defining the start region.
        end_region (list[tuple[int,int]]): List of (row, col) coordinates defining the end region.
        average (str): Either 'harmonic' or 'arithmetic'. Determines how to average conductivities
                       between neighboring sites.

    Returns:
        tuple:
            - path (np.ndarray): Optimal path as list of (row, col) coordinates.
            - total_resistance (np.longdouble): Total resistance along the optimal path.
            - actual_start_node (tuple[int,int]): The specific node where the path starts.
            - actual_end_node (tuple[int,int]): The specific node where the path ends.
            - resistance_map (np.ndarray): 2D array of minimal cumulative resistance.
    """

    Lx, Ly = K.shape
    pq = []
    resist = {}
    predecessors = {}
    visited = set()

    # --- Initialize all nodes with infinite resistance ---
    for r in range(Lx):
        for c in range(Ly):
            resist[(r, c)] = np.longdouble(np.inf)

    # --- Start region initialization ---
    for r, c in start_region:
        resist[(r, c)] = 0.0
        heapq.heappush(pq, (0.0, r, c))

    # --- 8-neighbor moves ---
    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),     # N, S, W, E
        (-1, -1), (-1, 1), (1, -1), (1, 1)    # NW, NE, SW, SE
    ]

    actual_end_node = None

    while pq:
        current_res, r, c = heapq.heappop(pq)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # --- Check if reached end region ---
        if (r, c) in end_region:
            actual_end_node = (r, c)
            break

        # --- Loop over neighbors ---
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < Lx and 0 <= nc < Ly):
                continue

            k1, k2 = K[r, c], K[nr, nc]
            if k1 <= 0 or k2 <= 0:
                continue  # skip invalid or zero conductivity

            # --- Compute bond conductivity ---
            if average == 'harmonic':
                k_bond = 2.0 * k1 * k2 / (k1 + k2)
            elif average == 'arithmetic':
                k_bond = 0.5 * (k1 + k2)
            else:
                raise ValueError("average must be either 'harmonic' or 'arithmetic'")

            if k_bond <= 0:
                continue

            # --- Distance weighting (diagonals longer) ---
            length_factor = np.sqrt(2.0) if dr != 0 and dc != 0 else 1.0

            # --- Edge resistance = path length / bond conductivity ---
            edge_res = length_factor / k_bond
            new_res = current_res + edge_res

            if new_res < resist[(nr, nc)]:
                resist[(nr, nc)] = new_res
                predecessors[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_res, nr, nc))

    # --- Backtrack to reconstruct path ---
    path = []
    total_resistance = np.longdouble(np.inf)
    actual_start_node = None

    if actual_end_node is not None:
        total_resistance = resist[actual_end_node]
        node = actual_end_node
        while node is not None:
            path.append(node)
            if node in start_region:
                actual_start_node = node
                break
            node = predecessors.get(node)
        path.reverse()

    # --- Convert resistance dict to 2D array ---
    resistance_map = np.full_like(K, np.inf, dtype=np.longdouble)
    for (r, c), val in resist.items():
        resistance_map[r, c] = val

    return np.array(path)[:,[1,0]], total_resistance, actual_start_node, actual_end_node, resistance_map

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
    
### Main Execution

if __name__ == '__main__':
    main()