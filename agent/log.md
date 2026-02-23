# Agent Log

## 2026-02-23 22:55:53 UTC
- Added `scripts/run.py` as the new hybrid Monte Carlo runner using `simulation/` package APIs.
- Implemented workflow per realization: fBm logK generation, PETSc head solve, face-centered velocity, LRP, and fastest-path tracking.
- Added GPU RK4 task queue workers (multi-GPU capable), with CPU RK45 fallback when RK4 fails.
- Kept output structure compatible with the existing `simulation/main.py` style.
- Added CLI validation and syntax checks for `scripts/run.py`.
- Updated `README.md` with run instructions, CLI reference, and output layout.
