"""
Configuration management and argument parsing.
"""

import argparse
import os
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

def update_config_file(path_config_file, config_update):
    """Update the YAML configuration file with values from config_update."""
    try:
        import yaml
    except ImportError:
        raise ImportError("update_config_file requires PyYAML. Install with: pip install PyYAML") from None

    if not os.path.exists(path_config_file):
        config_dict = {}
    else:
        with open(path_config_file, 'r') as file:
            config_dict = yaml.safe_load(file) or {}

    update_config_dict(config_dict, config_update)

    with open(path_config_file, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

def update_config_dict(config_dict, config_update):
    '''Update config_dict with values from config_update, handling nested dictionaries and None values.'''
    for key, value in config_update.items():
        if isinstance(value, dict):  # If the value is a dictionary, call the function recursively
            if key not in config_dict:
                config_dict[key] = {} # If key does not exist, create it as an empty dictionary
            update_config_dict(config_dict[key], value)
        else:  # Update the value if not None
            if value is not None:
                config_dict[key] = value

def create_par2_config(dx, dy, lx, ly, path_mfoutput, path_p2output, source_vertex, x_cp):
    """Create PAR2 configuration dictionary from grid and path parameters."""
    return {
        'grid': {
            'cell size': [dx, dy, 1],
            'dimension': [lx, ly, 1]
        },
        'physics': {
            'porosity': 0.25,
            'molecular diffusion': 0.0,
            'longitudinal dispersivity': 0.0,
            'transverse dispersivity': 0.0,
            'velocity': {
                'type': 'modflow',
                'file': str(path_mfoutput / "mf2005_model_modified.ftl"),
            }
        },
        'simulation': {
            'particles': {
                'N': 1000,
                'start': {
                    'p1': source_vertex[0],
                    'p2': source_vertex[1]
                }
            },
            'dt': 0.2,
            'steps': 1000000
        },
        'output': {
            'output_dir': str(path_p2output),
            'debut': {
                'try_then_run': True,
                'save_try_traj': False,
                'save_spatial_moment': False,
                'traj_file_name': 'traj_*.csv',
                'skip': 1,
                'num': 5,
                'x': x_cp,
                'interval': 100,
            },
        },
    }


def create_lazymole_config(lx, ly, dx, dy, path_lminput, path_lmoutput):
    """Create lazymole configuration dictionary from grid and path parameters."""
    return {
        'grid': {
            'dimensions': {
                'nx': lx,
                'ny': ly,
                'nz': 1
            },
            'cell size': {
                'dx': dx,
                'dy': dy,
                'dz': 1
            },
            'refinement': {
                'refx': 1,
                'refy': 1,
                'refz': 1
            }
        },
        'input': {
            'field': {
                'file': str(path_lminput / "logk_field.dat"),
                'skip': 0,
                'log': True
            },
            'source': {
                'file': str(path_lminput / "source.dat")
            },
            'target': {
                'file': str(path_lminput / "target.dat")
            }
        },
        'output': {
            'resistance': {
                'file': str(path_lmoutput / "resistance_map.dat")
            },
            'path': {
                'file': str(path_lmoutput / "least_resistance_path.dat")
            }
        }
    }
