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
- Backward compatibility: default remains `velocity_location="cell"`, so existing callers continue to work without changes.
