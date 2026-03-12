"""
Particle tracking: CPU (RK45), Pollock, PAR2 (external), and GPU (CUDA) backends.

Use this package as before:
  from simulation import particle_tracking
  particle_tracking.particle_tracking_cpu(params)
  particle_tracking.particle_tracking_pollock(params)
  particle_tracking.fastest_trajectory(px, py, pt, L, st)
  particle_tracking.run_par2(path_par2, path_p2input, path_p2output)
  particle_tracking.gpu_track_particles_arclength(...)

Or import submodules explicitly:
  from simulation.particle_tracking import rk45_cpu, pollock, par2, rk45_gpu
"""

from .rk45_cpu import (
    find_le_index,
    bilinear_sample,
    norm2,
    rhs_arclength,
    rk45_one_particle,
    particle_tracking_cpu,
)
from .pollock import (
    pollock_step,
    trace_pathline,
    particle_tracking_pollock,
)
from .par2 import run_par2
from .utils import fastest_trajectory, fastest_trajectory_with_snapshot

try:
    # Numba's CUDA import can fail at import time if a driver isn't present/configured.
    from .rk45_gpu import gpu_track_particles_arclength  # noqa: F401
    _gpu_available = True
except Exception:
    gpu_track_particles_arclength = None
    _gpu_available = False

__all__ = [
    "find_le_index",
    "bilinear_sample",
    "norm2",
    "rhs_arclength",
    "rk45_one_particle",
    "particle_tracking_cpu",
    "pollock_step",
    "trace_pathline",
    "particle_tracking_pollock",
    "fastest_trajectory",
    "fastest_trajectory_with_snapshot",
    "run_par2",
]

if _gpu_available:
    __all__.append("gpu_track_particles_arclength")
