"""
Particle tracking: CPU (RK45), PAR2 (external), and GPU (CUDA) backends.

Use this package as before:
  from simulation import particle_tracking
  particle_tracking.particle_tracking_cpu(params)
  particle_tracking.fastest_trajectory(px, py, pt, L, st)
  particle_tracking.run_par2(path_par2, path_p2input, path_p2output)
  particle_tracking.gpu_track_particles_arclength(...)

Or import submodules explicitly:
  from simulation.particle_tracking import rk45_cpu, par2, rk45_gpu
"""

from .rk45_cpu import (
    find_le_index,
    bilinear_sample,
    norm2,
    rhs_arclength,
    rk45_one_particle,
    particle_tracking_cpu,
)
from .par2 import run_par2
from .rk45_gpu import gpu_track_particles_arclength
from .utils import fastest_trajectory, fastest_trajectory_with_snapshot

__all__ = [
    "find_le_index",
    "bilinear_sample",
    "norm2",
    "rhs_arclength",
    "rk45_one_particle",
    "particle_tracking_cpu",
    "fastest_trajectory",
    "fastest_trajectory_with_snapshot",
    "run_par2",
    "gpu_track_particles_arclength",
]
