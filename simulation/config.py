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

def create_par2_config(dx, dy, lx, ly, path_mfoutput, path_p2output, source_vertex, x_cp, overrides=None):
    """Create PAR2 configuration dictionary from grid and path parameters.

    Optional overrides: a dict of key-value pairs (nested keys supported) merged into
    the base config, so any PAR2 YAML key can be adjusted (e.g. simulation.dt,
    simulation.particles.N, output.debut.num) without changing this function.
    """
    config = {
        'grid': {
            'cell size': [dx, dy, 1],  # Size of each grid cell in x, y, z directions
            'dimension': [lx, ly, 1]     # Number of cells in x, y, z directions
        },
        'physics': {
            'porosity': 0.25,                   # Porosity of the medium (fraction of the volume that is void space)
            'molecular diffusion': 0.0, #1.0e-6,     # Molecular diffusion coefficient
            'longitudinal dispersivity': 0.0, #0.01,  # Longitudinal dispersivity (how particles spread along the flow direction)
            'transverse dispersivity': 0.0, #0.001,   # Transverse dispersivity (how particles spread perpendicular to flow)
            'velocity': {
                'type': 'modflow',                       # Type of the velocity data file (Modflow format)
                'file': str(path_mfoutput / "mf2005_model_modified.ftl"),  # Path to the file containing velocity data
            }
        },
        'simulation': {
            'particles': {
                'N': 10000,  # Number of particles to simulate
                'start': { # Define the volume where the particles are initially located p1x <= x <= p2x, p1y <= y <= p2y, p1z <= z <= p2z
                    'p1': source_vertex[0],  # Starting position of particles (lower bounds)
                    'p2': source_vertex[1]   # Ending position of particles (upper bounds)
                }
            },
            'dt': 0.25,  # Time step size
            'steps': 100000  # Number of simulation steps to run
        },
        'output': {
            'output_dir': str(path_p2output),
            # 'csv': {
            #     'file': './par2_output/result.csv',  # Path to the output CSV file for mc_result
            #     'skip': 10000,  # Data is computed every 'skip' steps
            #     'items': [  # List of items to be computed. You can add any number of items here
            #         {'label': 'cbtx', 'type': 'after-x', 'x': x_cp},  # Compute fraction of particles that have crossed the plane x=x_cp
            #         # {'label': 'Inside Box', 'type': 'box', 'p1': target_vertex[0], 'p2': target_vertex[1]},  # Compute fraction of particles inside the box
            #     ]
            # },
            'snapshot': {
                'file': str(path_p2output / 'snapshots/snap_*.csv'),  # Path to the snapshot output files. A file is generated for each output step. Note that exporting a large number of snapshots may slow down the simulation.
                'skip': 100,  # Save snapshots every 'skip' steps, or use 'steps' to specify exact steps
                # 'steps': [0, 2500, 5000, 7500, 10000], # List of time steps to save snapshots
            },
            'debut': {
                'try_then_run': True, # Determine the algorithm to detect debut trajectories. True: Run without saving trajectories, then narrow the release box around the first debut particle, and rerun to detect the approximate debut trajectory. False: Record all trajectories and detect the exact debut trajectory.
                'save_try_traj': False, # True: Save particle trajectories in the try phase. False: Not initialize traj vectors to hugely save memory.
                'save_spatial_moment': False, # True: Save spatial moments of particle trajectories in the try phase. 
                'traj_file_name': 'traj_*.csv',  # Path to the debut output files. A file is generated for each debut trajectory.
                'skip': 1,  # Data is computed every 'skip' steps
                'num': 100,  # Number of debut particles to be tracked
                'x': x_cp,  # X coordinate of the control plane
                'interval': 100,  # Trajectory export interval from GPU to CPU, in number of time steps
            },
        },
    }
    
    if overrides is not None:
        update_config_dict(config, overrides)
    return config


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
