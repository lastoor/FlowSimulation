# Agent Log

## 2026-03-03 04:00:53 UTC
- Added RK45 CPU support for `velocity_location` in `("cell", "face")`, including face-based coordinate-wise interpolation.
- Added RK45 GPU support for the same `velocity_location` behavior, plus optional `x_cell_centers`/`y_cell_centers`.
- Kept backward-compatible exported CPU helper signatures (`rhs_arclength`, `rk45_one_particle`) for existing imports.
- Added CPU particle-tracking tests for face-mode uniform flow and invalid `velocity_location` handling.
- Updated `README.md` with velocity-location conventions and RK45 face/cell usage notes.

## 2026-02-23 22:55:53 UTC
- Added `scripts/run.py` as the new hybrid Monte Carlo runner using `simulation/` package APIs.
- Implemented workflow per realization: fBm logK generation, PETSc head solve, face-centered velocity, LRP, and fastest-path tracking.
- Added GPU RK4 task queue workers (multi-GPU capable), with CPU RK45 fallback when RK4 fails.
- Kept output structure compatible with the existing `simulation/main.py` style.
- Added CLI validation and syntax checks for `scripts/run.py`.
- Updated `README.md` with run instructions, CLI reference, and output layout.
