"""
Configuration management and argument parsing.
"""

import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_workdir', required=True, type=str,
                       help="Path to the working directory where flow packages lie in: str")
    parser.add_argument('--num_mc', required=True, nargs='+', type=int,
                       help='Number of MC: int list')
    parser.add_argument('--field_size', required=True, nargs='+', type=int,
                       help='Field size: int list')
    parser.add_argument('--hurst', required=True, type=float,
                       help="Hurst exponent")
    parser.add_argument('--minpadding', type=int, default=0,
                       help="Minimal padding for fBm generation")
    parser.add_argument('--c', type=float, default=None,
                       help="Coefficient for semivariogram for fBm generation")
    parser.add_argument('--std', type=float, default=None,
                       help="Standard deviation for fBm generation")
    return parser.parse_args()


def create_base_config(args):
    """Create base configuration dictionary from parsed arguments."""
    path_workdir = Path(args.path_workdir).expanduser().resolve()

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
    config_base['path_mainlogfile'] = config_base['path_log'] / 'simulation_{timestamp}.log'
    config_base['path_gpulogfile'] = config_base['path_log'] / 'gpu_profile_{timestamp}.log'
    config_base['path_cpulogfile'] = config_base['path_log'] / 'cpu_profile_{timestamp}.log'

    return config_base