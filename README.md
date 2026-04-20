# FlowSimulation

FlowSimulation is a Python-based workflow for 2D flow simulation and trajectory analysis on heterogeneous conductivity fields.

## What it does
- Generates log-hydraulic conductivity fields with fractional Brownian motion (fBm)
- Solves hydraulic head using PETSc
- Computes face-centered Darcy velocity fields
- Computes least-resistance paths (LRP) with Dijkstra
- Computes fastest paths (FP) with particle tracking:
  - RK4 FP (Taichi) with cell/face velocity interpretation
  - RK45 on CPU/GPU with cell/face velocity interpretation
  - Pollock particle tracking on CPU with full face-centered velocity fields
- Runs Monte Carlo simulations in parallel across CPUs and GPUs

## Repository structure
- `simulation/`: core package (generation, flow solve, path finding, particle tracking)
- `scripts/run_gpu.py`: legacy monolithic runner (ancestor)
- `scripts/run.py`: current hybrid CPU+GPU Monte Carlo runner
- `tests/`: comparison and validation workflows

## Requirements
Typical runtime dependencies include:
- `numpy`, `scipy`
- `petsc4py`
- `numba`
- `taichi` (for RK4 GPU and Pollock GPU summary paths)
- Optional: CUDA-capable environment and `nvidia-smi`

Install the repository requirements with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes the core runtime packages plus comparison / visualization dependencies that are imported by scripts and tests. GPU-oriented packages such as `cupy` and `taichi` are included for environments that use those paths.

## Run the new hybrid script
Use exactly one of `--std` or `--c`.

Zero-variance behavior:
- `--std 0` is valid and produces a uniform zero `logk_field`.
- `--c 0` is also valid and produces the same uniform zero `logk_field`.
- Downstream, `k_field = exp(logk_field)` is therefore uniform at `1.0` for this default zero-mean fBm path.

### Single size example
```bash
python scripts/run.py \
  --path_workdir /home1/binhaoli/FlowSimulation/output \
  --num_mc 10 \
  --field_size 100 \
  --hurst 0.25 \
  --std 6
```

### Multi-size example
```bash
python scripts/run.py \
  --path_workdir /home1/binhaoli/FlowSimulation/output \
  --num_mc 10 5 \
  --field_size 100 200 \
  --hurst 0.25 \
  --std 6 \
  --cpu_workers 8 \
  --gpu_workers 2 \
  --seed 123 \
  --num_trial 50
```

## CLI reference (`scripts/run.py`)
- `--path_workdir` (required): output root
- `--num_mc` (required): list of MC counts (same length as `--field_size`)
- `--field_size` (required): list of grid sizes
- `--hurst` (required): Hurst exponent
- `--minpadding`: fBm embedding padding (default `0`)
- `--std` or `--c`: choose one to parameterize fBm scale
- `--cpu_workers`: max CPU worker processes
- `--gpu_workers`: max GPU worker processes (auto-detect visible GPUs if omitted)
- `--seed`: base seed for reproducibility
- `--num_trial`: max retries per MC realization

## Output layout
For each field size `L`, outputs are written under:
- `hurst{H}_size{L}/main_output`
- `hurst{H}_size{L}/path`
- `hurst{H}_size{L}/logk_field`

Main summary file per size:
- `main_output/mainOutput_size{L}.npz`

Per-MC file:
- `main_output/mainOutput_size{L}_mcXXXX.npz`

## Notes
- If GPU RK4 tracking fails on a realization, the workflow falls back to CPU RK45 tracking.
- PETSc solve failures or invalid trajectories trigger retry logic up to `--num_trial`.
- For generic Gaussian random fields built directly from `GaussianRandomFieldCircEmbed`, using `ExpCov(sigma=0)` returns the configured mean field exactly.

## Particle-Tracking Summary API
- `simulation.particle_tracking.particle_tracking_summary(params)` is the high-level entrypoint for travel-time-oriented particle tracking.
- New common options:
  - `backend`: `"cpu"` or `"gpu"`
  - `method`: `"pollock"`, `"rk45"`, or `"rk4"`
  - `tracking_mode`: `"trajectory"` or `"arrival"`
  - `stop_mode`: `"first"`, `"fraction"`, or `"full_btc"`
  - `arrival_fraction`: fraction of released particles for `stop_mode="fraction"`
  - `save_trajectories`: whether to attach the selected trajectories to the summary output
  - `return_fastest`: whether to include the fastest-path payload directly in the returned summary
- `particle_tracking_summary(params)` is now the canonical high-level API for both fastest-path and arrival/BTC workflows.
- The existing low-level APIs such as `particle_tracking_cpu(...)` and `fastest_trajectory(...)` remain unchanged for compatibility, but new code should prefer the summary API.
- Current method/backend support:
  - `method="pollock"` with `backend="cpu"` or `backend="gpu"`
  - `method="rk45"` with `backend="cpu"` or `backend="gpu"`
  - `method="rk4"` with `backend="gpu"`
- If `method` is omitted, the summary API defaults to `rk45` on CPU and `rk4` on GPU.
- The GPU summary paths for `rk4` and `pollock` use Taichi and are optimized for arrival metrics; trajectory replay is currently limited to selected particles and not the full breakthrough-curve population.
- The main simulation/runtime flow now consumes the summary result directly instead of tracking everything and then calling a separate fastest-path reduction helper.

## Velocity Field Location Convention
- `velocity_location="cell"`: `vx`/`vy` are treated as cell-centered and interpolated bilinearly.
- `velocity_location="face"`: `vx`/`vy` are treated with coordinate-wise face interpolation (`vx` along x-faces, `vy` along y-faces), consistent with `simulation/particle_tracking/rk4_fp.py`.
- In face mode, `x` and `y` are interpreted as cell-center coordinates; face coordinates are derived by midpoint/extrapolation.
- RK45 optional face-mode inputs:
  - `x_cell_centers`
  - `y_cell_centers`
  - if omitted, RK45 uses `x` and `y` as the center arrays.
- RK45 CPU/GPU now use the full Dormand-Prince embedded pair for error control, including the final `k7` stage used by the 4th-order estimate.
- Accepted RK45 steps that cross `right_x` are now clipped against the pre-step state, so the recorded endpoint and travel time land on the actual first crossing.
- Backward compatibility: default remains `velocity_location="cell"`, so existing callers continue to work without changes.

## Pollock Backend
- `simulation.particle_tracking.particle_tracking_pollock(params)` is available as a first-class particle-tracking backend.
- `simulation.particle_tracking.particle_tracking_summary(params)` also supports `method="pollock", backend="gpu"` for Taichi-based arrival/summary execution with optional replay of selected trajectories.
- The CPU implementation now lives in `simulation/particle_tracking/pollock_cpu.py`, with shared Pollock constants/input parsing in `simulation/particle_tracking/pollock_core.py`.
- Pollock requires explicit face coordinates and full staggered face velocities:
  - `x_faces`: shape `(nx + 1,)`
  - `y_faces`: shape `(ny + 1,)`
  - `vx`: shape `(ny, nx + 1)`
  - `vy`: shape `(ny + 1, nx)`
- The expected velocity input matches `simulation.hydraulic.compute_velocity(..., output_location="face", face_output="full")`.
- Pollock returns the same batched trajectory contract as RK45: `px, py, pt, L, st`, so `fastest_trajectory(...)` works without adapters.
- Low-level helpers `pollock_step(...)` and `trace_pathline(...)` remain available for direct analytical tracing.
- CPU Pollock now splits arrival-only and trajectory modes internally: arrival summaries avoid dense `(n_particles, max_points)` path storage, while the legacy tuple API still returns full paths.
- GPU Pollock now tracks each particle to completion inside a single Taichi kernel launch instead of looping step-by-step from Python.
- Pollock capture at `right_x` is now handled analytically inside the per-cell step, so particles stop exactly on an in-cell target boundary instead of being linearly clipped afterward.
- If Taichi GPU initialization fails, Pollock GPU now emits a warning before falling back to the Taichi CPU backend.
- The Taichi Pollock helper functions have also been refactored to avoid unsupported early returns inside `@ti.func`, which is required for Taichi 1.6 compatibility.
- GPU Pollock summary results now mask non-success particle times before arrival ordering / fastest-particle selection, so failed particles cannot be incorrectly reported as zero-time “fastest” arrivals.
- GPU Pollock summary now also supports snapshot capture through `snapshot_mode` / `snapshot_times`; physical snapshot times are handled in one pass, while `dt_interval` and `fraction_of_first_arrival` use a first pass to resolve absolute times before rerunning with snapshot buffers attached.

Example:
```python
from simulation import hydraulic, particle_tracking

vx_face, vy_face = hydraulic.compute_velocity(
    K, x_centers, y_centers, h,
    output_location="face",
    face_output="full",
)

params = {
    "x_faces": x_faces,
    "y_faces": y_faces,
    "vx": vx_face,
    "vy": vy_face,
    "seeds_x": seeds_x,
    "seeds_y": seeds_y,
    "right_x": x_faces[-1],
}
px, py, pt, L, st = particle_tracking.particle_tracking_pollock(params)
best = particle_tracking.fastest_trajectory(px, py, pt, L, st)
```

## PAR2 Comparison Workflow
- `tests/Compare_PAR2_Py/compare.py` now includes Pollock CPU and Pollock GPU as separate solver entries alongside PAR2, RK45 CPU/GPU, RK4 FP, and optional MODPATH7.
- New CLI control:
  - `--no-pollock`: skip both Pollock backends while leaving the other solvers enabled.
  - `--no-pollock-cpu`: skip only Pollock CPU.
  - `--no-pollock-gpu`: skip only Pollock GPU.
- The comparison script still uses trimmed `QXX`/`QYY`-style arrays for PAR2/RK45/RK4 plotting and file outputs, but it now reconstructs Pollock inputs internally:
  - `x_faces`, `y_faces`
  - `vx_full (ny, nx + 1)`
  - `vy_full (ny + 1, nx)`
- Pollock CPU and Pollock GPU now run through `particle_tracking_summary(...)` and save/plot their fastest-path results independently.
- Pollock results are now included in:
  - console travel-time summaries as `Pollock CPU` and `Pollock GPU`
  - saved `.npz` outputs as `pollock_cpu_x`, `pollock_cpu_y`, `pollock_cpu_time`, `pollock_gpu_x`, `pollock_gpu_y`, `pollock_gpu_time`
  - saved reconstructed inputs as `pollock_x_faces`, `pollock_y_faces`, `pollock_vx_full`, `pollock_vy_full`
  - figure overlays and synthetic-case evaluation metrics
- Example:

```bash
python tests/Compare_PAR2_Py/compare.py --test-case synthetic --no-par2 --no-rk45-gpu
```

## Hydraulic Grid Options
- `simulation.hydraulic.get_spacing_and_faces(coord_centers)` is the shared helper used to infer cell widths and face spacings from 1D cell-center coordinates.
- `solve_head(..., spacing_mode="uniform")` keeps the current live solver behavior with uniform `dx=1/nx`, `dy=1/ny`.
- `solve_head(..., spacing_mode="center_faces")` uses `get_spacing_and_faces(x)` and `get_spacing_and_faces(y)` for coordinate-derived spacing in PETSc assembly.
- `compute_velocity(..., output_location="face", face_output="trimmed")` returns per-cell right/top-face arrays with shape `(ny, nx)`.
- `compute_velocity(..., output_location="face", face_output="full")` returns staggered full face arrays `vx_face (ny, nx+1)` and `vy_face (ny+1, nx)`.
- `compute_velocity(..., align_with_modflow=True)` zeros the outer right/top faces to match MODFLOW `QXX/QYY`; `False` preserves the computed boundary face fluxes.
- `compute_velocity(..., output_location="cell")` now averages the full left/right and bottom/top face arrays for every cell.

## MODFLOW vs PETSc Comparison CLI
- `tests/Compare_Modflow_PETSc/test_compare_modflow_petsc.py` can now be run directly with CLI options instead of only through `unittest`.
- Supported runtime controls include:
  - `--num-pt`
  - `--dh`
  - `--rtol`, `--atol`, `--max-it`
  - `--output-location {face,cell}`
  - `--align-with-modflow-velocity` / `--no-align-with-modflow-velocity`
  - `--spacing-mode {uniform,center_faces}`
  - `--face-output {trimmed,full}`
  - `--skip-figures`
  - `--out-dir`
- Default direct-run behavior is kept compatible with the old test path: `output_location="face"`, `align_with_modflow=True`, `spacing_mode="uniform"`, and `face_output="trimmed"`.
- Example:

```bash
python tests/Compare_Modflow_PETSc/test_compare_modflow_petsc.py \
  --dh 0.02 \
  --spacing-mode center_faces \
  --output-location face \
  --face-output full
```

## Possible issues
- `requirements.txt` includes optional GPU-oriented packages such as `cupy` and `taichi`; these may require environment-specific installation steps on machines without compatible GPU support.

1. `solve_head` still defaults to `spacing_mode="uniform"` for backward compatibility; callers must opt in to `spacing_mode="center_faces"` to use coordinate-derived spacing.
2. Boundary gradients in `compute_velocity` are still one-sided; higher-order boundary discretization remains a separate improvement.
3. Particle-tracking solvers do not yet share one unified velocity-input abstraction: RK45 CPU/GPU support both cell-centered and face-centered velocity interpretation, RK4/Taichi also supports both modes through its own packed velocity-field representation, while Pollock is face-only and requires explicit full staggered face arrays.
4. GPU particle tracking is currently split across two stacks: RK45 GPU uses `numba.cuda`, while RK4 GPU uses `taichi`. This means different runtime dependencies, initialization behavior, debugging workflows, and likely different performance/portability characteristics.
5. The method/backend support matrix is still asymmetric: `pollock` now supports CPU and GPU, `rk45` supports CPU and GPU, but `rk4` is still exposed as GPU-only. A longer-term cleanup would make CPU/GPU availability consistent across all three methods.
6. Pollock’s public API is still more specialized than RK45/RK4: it consumes `x_faces`, `y_faces`, and full staggered face velocities directly, while RK45/RK4 use the more generic `x/y + vx/vy (+ velocity_location)` interface. A shared adapter layer would reduce this split.
7. `tracking_mode="arrival"` is now implemented efficiently for Pollock CPU/GPU, but not all underlying implementations are fully optimized for arrival-only workloads yet. In particular, the current RK45 GPU summary path still wraps the existing full-trajectory CUDA kernel, so memory use is not yet minimized there.
8. GPU RK4/Taichi and GPU Pollock/Taichi trajectory replay are still limited compared with the CPU tuple-based paths: they are currently intended for selected trajectories rather than full-trajectory retention for the entire particle population.
9. The end-to-end Monte Carlo flow in `simulation/main.py` still assumes a fastest-trajectory workflow for FP geometry checks. The new arrival/fraction/full-BTC summary modes exist, but the main simulation loop is not yet fully redesigned around breakthrough-curve analysis.
10. The legacy tuple-returning particle-tracking APIs and `fastest_trajectory(...)` helper still exist for compatibility, so there is still some duplication in the public surface even though the live runtime flow now uses the direct-return summary path.
11. Pollock still treats “captured at `right_x`” and “exited the domain through another boundary” under the same success status code. Arrival summaries still use travel-time/selection logic correctly, but a future cleanup could separate those outcomes more explicitly.
12. Pollock accuracy can degrade on highly oscillatory velocity fields because it assumes approximately linear velocity variation within each cell. A planned future improvement is “virtual subcell Pollock”: subdivide each coarse cell into smaller physical subcells, resample the staggered face velocities onto the refined grid, and then run the existing Pollock solver on that refined field. For strongly sub-cell-varying fields, RK45/RK4 should still be treated as the more reliable reference family.
