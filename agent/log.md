# Agent Log

## 2026-03-13 01:20:00 UTC
- Split Pollock into a shared core plus explicit CPU/GPU backend modules: added `simulation/particle_tracking/pollock_core.py`, moved the CPU implementation into `simulation/particle_tracking/pollock_cpu.py`, and kept `simulation/particle_tracking/pollock.py` as a compatibility shim.
- Centralized Pollock status codes, numerical thresholds, and common input parsing so the CPU and GPU backends now share one source of truth for those behaviors.
- Updated GPU Pollock to warn when Taichi GPU initialization fails and it falls back to the Taichi CPU backend.
- Fixed GPU Pollock tracker reruns by restoring particle positions from stored seed fields during `init_state()`.
- Added particle-tracking tests for Taichi fallback warnings, cached GPU tracker reruns, and explicit near-zero Pollock stagnation behavior.
- Reverified the particle-tracking test module with `python tests/test_particle_tracking.py` and rechecked the touched modules with `python -m py_compile`.
- Updated `README.md` to document the new Pollock module layout, explicit Taichi fallback warning, and the remaining Pollock status-code limitation.

## 2026-03-13 01:28:00 UTC
- Removed the now-unneeded `simulation/particle_tracking/pollock.py` compatibility shim after confirming the repo no longer imports that submodule directly.
- Kept the public package-level Pollock API unchanged through `simulation/particle_tracking/__init__.py`, which already exports the CPU backend from `pollock_cpu.py`.
- Updated `README.md` so the documented Pollock module layout no longer mentions the deleted shim.

## 2026-03-13 00:55:00 UTC
- Optimized Pollock CPU tracking by splitting arrival-only and dense-trajectory execution paths and parallelizing both with Numba `prange`.
- Removed the old Python-driven Taichi step loop from GPU Pollock and moved full per-particle tracking into a single kernel launch.
- Changed Pollock CPU/GPU analytical stepping so forward exits target `min(x_face, right_x)` directly, eliminating the outer-loop linear clipping hack and making in-cell `right_x` capture exact.
- Kept the public summary API unchanged while making Pollock arrival mode lighter-weight internally and retaining replay-only trajectory reconstruction for selected particles.
- Added Pollock regression tests for exact in-cell `right_x` capture, arrival-only summary mode without dense arrays, and GPU Pollock capture parity.
- Reverified the particle-tracking test module with `python tests/test_particle_tracking.py`.
- Updated `README.md` to document the new Pollock execution behavior and narrowed the remaining arrival-mode limitation note to the still-unoptimized RK45 GPU path.

## 2026-03-13 00:25:00 UTC
- Made `particle_tracking_summary(params)` the canonical high-level particle-tracking return path for the live runtime flow.
- Added `return_fastest` to the summary API so particle tracking can directly include or omit the fastest-path payload without a separate reduction step.
- Centralized fastest-result construction in shared particle-tracking utilities and switched the summary backends to use it consistently.
- Removed the remaining legacy `particle_tracking_* -> fastest_trajectory(...)` reduction path from `simulation/main.py` and `scripts/run.py`.
- Kept tuple-returning trackers and `fastest_trajectory(...)` as compatibility helpers rather than removing them outright.
- Added tests covering `return_fastest=False` and reverified the particle-tracking test module with `python tests/test_particle_tracking.py`.
- Updated `README.md` to document the canonical summary API and compatibility status of the older reduction helper flow.

## 2026-03-13 00:05:00 UTC
- Added a Taichi-based GPU Pollock summary backend under `method="pollock", backend="gpu"`.
- Kept GPU Pollock summary-first: it computes arrival metrics on GPU and replays only selected trajectories with the existing Pollock stepping logic.
- Updated the method-aware dispatcher and package exports so Pollock now supports both CPU and GPU in the high-level summary API.
- Added particle-tracking tests covering GPU Pollock arrival summaries and selected-trajectory replay.
- Verified the updated particle-tracking test module with `python tests/test_particle_tracking.py`.
- Updated `README.md` with the new Pollock GPU support and the revised support-matrix/possible-issues notes.

## 2026-03-12 23:05:00 UTC
- Added `method="pollock" | "rk45" | "rk4"` to the high-level `particle_tracking_summary(params)` API.
- Added CPU Pollock summary support and GPU RK45 summary support alongside the existing CPU RK45 and GPU RK4/Taichi summary paths.
- Made the dispatcher validate the current support matrix explicitly: Pollock on CPU, RK45 on CPU/GPU, RK4 on GPU.
- Added `method` to particle-tracking config defaults and wired Pollock-compatible face inputs into `simulation/main.py`.
- Added particle-tracking tests for Pollock CPU summaries, RK45 GPU summaries, backend-default method selection, and invalid method/backend combinations.
- Verified the updated particle-tracking test module with `python tests/test_particle_tracking.py`.
- Updated `README.md` with the new `method` selector and the current method/backend support table.

## 2026-03-12 22:10:00 UTC
- Added `simulation.particle_tracking.particle_tracking_summary(params)` as a high-level particle-tracking API with `backend`, `tracking_mode`, `stop_mode`, `arrival_fraction`, and `save_trajectories` controls.
- Added shared arrival-analysis helpers for per-particle travel times, arrival ordering, threshold stopping, and selected trajectory extraction.
- Added CPU RK45 summary support while preserving the existing tuple-returning `particle_tracking_cpu(...)` and `fastest_trajectory(...)` APIs.
- Added Taichi-based GPU summary support for arrival metrics, with optional replay of selected trajectories and current replay limits documented for non-`first` stopping modes.
- Added particle-tracking tests covering arrival summaries, CPU fraction stopping, and a Taichi-backed GPU arrival-mode smoke path.
- Verified the updated particle-tracking test module with `python tests/test_particle_tracking.py`.
- Updated `README.md` with the new particle-tracking summary API and backend/stopping-mode notes.

## 2026-03-12 20:15:00 UTC
- Added deterministic zero-variance handling in `simulation/grf_generation.py` for degenerate random fields.
- Updated `compute_c(..., std_target=0)` to return `0.0` and made `fBmGenerator(c=0, ...)` return a uniform zero `logk_field`.
- Updated `ExpCov(sigma=0, ...)` and `GaussianRandomFieldCircEmbed(...)` so zero covariance scale returns the configured mean field instead of sampling through the FFT path.
- Added GRF tests covering zero `std`, zero `c`, zero `sigma`, and zero-scale `generate_logk_field(...)`.
- Verified the GRF test module with `python tests/test_grf_generation.py`.
- Updated `README.md` to document the zero-variance uniform-field behavior.

## 2026-03-12 03:20:00 UTC
- Added Pollock support to `tests/Compare_PAR2_Py/compare.py` as a first-class comparison backend alongside PAR2, RK45 CPU/GPU, RK4 FP, and MODPATH7.
- Added the `--no-pollock` CLI flag and included Pollock in console summaries, saved `.npz` outputs, figure overlays, and synthetic-case evaluation.
- Added internal reconstruction of full Pollock face inputs from the script's existing trimmed `QXX`/`QYY`-style arrays.
- Verified the updated comparison script with a compile check and CLI help output.
- Updated `README.md` with the Pollock comparison workflow and `compare.py` usage notes.

## 2026-03-12 02:58:23 UTC
- Refactored `tests/Compare_Modflow_PETSc/test_compare_modflow_petsc.py` into a shared runner plus `argparse` CLI for direct execution.
- Added CLI controls for grid size, `dh`, PETSc tolerances, `output_location`, `align_with_modflow`, `spacing_mode`, `face_output`, figure saving, and output directory.
- Kept the existing `unittest` path on the old default behavior by routing it through the same runner with default options.
- Verified the new script syntax and CLI help output.
- Updated `README.md` with the new MODFLOW vs PETSc comparison CLI usage.

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
