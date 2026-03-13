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
  from simulation.particle_tracking import rk45_cpu, pollock_cpu, pollock_gpu, par2, rk45_gpu
"""

from .rk45_cpu import (
    find_le_index,
    bilinear_sample,
    norm2,
    rhs_arclength,
    rk45_one_particle,
    particle_tracking_cpu,
    particle_tracking_cpu_summary,
)
from .pollock_cpu import (
    pollock_step,
    trace_pathline,
    particle_tracking_pollock,
    particle_tracking_pollock_summary,
)
from .pollock_gpu import particle_tracking_pollock_gpu_summary
from .par2 import run_par2
from .utils import (
    compute_travel_times,
    extract_particle_trajectories,
    fastest_trajectory,
    fastest_trajectory_with_snapshot,
    summarize_arrivals,
)
from .api import particle_tracking_rk45_gpu_summary, particle_tracking_summary

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
    "particle_tracking_cpu_summary",
    "pollock_step",
    "trace_pathline",
    "particle_tracking_pollock",
    "particle_tracking_pollock_summary",
    "particle_tracking_pollock_gpu_summary",
    "particle_tracking_rk45_gpu_summary",
    "particle_tracking_summary",
    "compute_travel_times",
    "extract_particle_trajectories",
    "fastest_trajectory",
    "fastest_trajectory_with_snapshot",
    "summarize_arrivals",
    "run_par2",
]

if _gpu_available:
    __all__.append("gpu_track_particles_arclength")
