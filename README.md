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
- `taichi` (for RK4 GPU path in `scripts/run.py`)
- Optional: CUDA-capable environment and `nvidia-smi`

## Run the new hybrid script
Use exactly one of `--std` or `--c`.

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
- Pollock requires explicit face coordinates and full staggered face velocities:
  - `x_faces`: shape `(nx + 1,)`
  - `y_faces`: shape `(ny + 1,)`
  - `vx`: shape `(ny, nx + 1)`
  - `vy`: shape `(ny + 1, nx)`
- The expected velocity input matches `simulation.hydraulic.compute_velocity(..., output_location="face", face_output="full")`.
- Pollock returns the same batched trajectory contract as RK45: `px, py, pt, L, st`, so `fastest_trajectory(...)` works without adapters.
- Low-level helpers `pollock_step(...)` and `trace_pathline(...)` remain available for direct analytical tracing.

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
- `tests/Compare_PAR2_Py/compare.py` now includes Pollock in the solver comparison workflow alongside PAR2, RK45 CPU/GPU, RK4 FP, and optional MODPATH7.
- New CLI control:
  - `--no-pollock`: skip Pollock while leaving the other solvers enabled.
- The comparison script still uses trimmed `QXX`/`QYY`-style arrays for PAR2/RK45/RK4 plotting and file outputs, but it now reconstructs Pollock inputs internally:
  - `x_faces`, `y_faces`
  - `vx_full (ny, nx + 1)`
  - `vy_full (ny + 1, nx)`
- Pollock results are now included in:
  - console travel-time summaries
  - saved `.npz` outputs as `pollock_x`, `pollock_y`, `pollock_time`
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

1. `solve_head` still defaults to `spacing_mode="uniform"` for backward compatibility; callers must opt in to `spacing_mode="center_faces"` to use coordinate-derived spacing.
2. Boundary gradients in `compute_velocity` are still one-sided; higher-order boundary discretization remains a separate improvement.
