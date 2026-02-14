"""
Utility functions for logging, profiling, and GPU management.
"""

import logging
import subprocess
import time
import os
import gc

import numpy as np

# Optional GPU imports
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    _cupy_available = False


def get_logger(name='simulation', path_logfile='simulation.log'):
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s'
        )
        file_handler = logging.FileHandler(path_logfile, mode='a')
        console_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def get_gpu_usage():
    """Query GPU memory usage via nvidia-smi."""
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


def euclidean_length(trajectory):
    """
    Calculate the Euclidean length of a trajectory using CuPy for GPU acceleration.
    Args:
        trajectory (np.ndarray): array of shape (N, D)
    Returns:
        float: Euclidean length of the trajectory.
    """
    # Select backend
    use_gpu = False
    if _cupy_available:
        try:
            cp.cuda.runtime.getDevice()  # test GPU availability
            use_gpu = True
            xp = cp
        except Exception:
            xp = np
    else:
        xp = np

    # Move data to chosen backend
    trajectory_xp = xp.asarray(trajectory)

    # Compute differences and length
    diff = trajectory_xp[1:] - trajectory_xp[:-1]
    length = xp.sum(xp.sqrt(xp.sum(diff ** 2, axis=-1)))

    # Convert back to NumPy float
    if use_gpu:
        length_np = cp.asnumpy(length)
    else:
        length_np = float(length)

    # Cleanup
    del trajectory_xp, diff, length
    gc.collect()

    return float(length_np)