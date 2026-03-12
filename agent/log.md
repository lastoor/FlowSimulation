# Agent Log

## 2026-03-12 02:45:48 UTC
- Fixed RK45 CPU and GPU Dormand-Prince error estimation to evaluate the missing 7th stage at the 5th-order solution before forming the embedded 4th-order estimate.
- Corrected accepted-step `right_x` crossing handling so CPU/GPU interpolate from the pre-step state instead of the overwritten accepted state.
- Added a CPU particle-tracking regression test for accepted-step overshoot/crossing interpolation.
- Updated `README.md` to document the corrected RK45 error-control and first-crossing behavior.

## 2026-03-12 03:08:00 UTC
- Added `simulation/particle_tracking/pollock.py` as a packaged Pollock particle-tracking backend using full face-centered velocities.
- Exposed `particle_tracking_pollock(...)`, `pollock_step(...)`, and `trace_pathline(...)` from `simulation.particle_tracking`.
- Standardized Pollock output to the existing batched trajectory contract `px, py, pt, L, st` so `fastest_trajectory(...)` can consume it directly.
- Added particle-tracking tests covering Pollock uniform-flow stepping, path tracing, stagnation, input validation, and package import/export.
- Updated `README.md` with Pollock backend usage notes and its dependency on `hydraulic.compute_velocity(..., face_output="full")`.

## 2026-03-12 02:34:44 UTC
- Added `hydraulic.get_spacing_and_faces(...)` as the shared cell-width and face-spacing helper.
- Added `solve_head(..., spacing_mode=...)` with backward-compatible `uniform` default and opt-in `center_faces` spacing from cell-center coordinates.
- Refactored `compute_velocity(...)` to use the shared spacing helper.
- Added `compute_velocity(..., face_output="trimmed" | "full")` to choose per-cell vs full staggered face output.
- Kept `align_with_modflow` support and updated cell-centered velocity output to average the full left/right and bottom/top faces.
- Added hydraulic tests for the shared helper, face output modes, alignment behavior, and `solve_head` spacing validation.

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
