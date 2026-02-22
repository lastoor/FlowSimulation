"""
Parcels JIT needs a working C compiler. If you see "cannot execute 'cc1'" from
conda's compiler, use the system compiler instead before running:
    export CC=gcc
    export CXX=g++
    python test.py

Velocity field: from fBm log-K -> PETSc head solve -> face-centered Darcy velocity.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from datetime import timedelta
import parcels

from simulation import grf_generation
from simulation import hydraulic

# ----------------------------
# 1) Generate log-K with fBm, solve for head with PETSc, get face-centered velocity
# ----------------------------
Lx, Ly = 101, 101
nx, ny = 101, 101
np.random.seed(42)

# Cell-center coordinates for hydraulic solver
x_cc = (np.arange(nx) + 0.5) * (Lx / nx)
y_cc = (np.arange(ny) + 0.5) * (Ly / ny)

# fBm log-K field (fBmGenerator gives shape (num_pt, num_pt) for dim=2)
num_pt = max(nx, ny)
fbm_gen = grf_generation.fBmGenerator(
    c=2.0,
    alpha=0.5,
    num_pt=num_pt,
    dim=2,
    minpadding=0,
)
logk_field, msg_k = grf_generation.generate_logk_field(fbm_gen)
if logk_field is None:
    raise RuntimeError(f"fBm generation failed: {msg_k}")
# Slice to domain shape (ny, nx)
logk_field = np.asarray(logk_field[:ny, :nx], dtype=np.float64)

K = np.exp(logk_field)

# PETSc solve for hydraulic head (Dirichlet left/right, no-flow top/bottom)
dh = 0.01
h, msg_h, is_success = hydraulic.solve_head(
    K, x_cc, y_cc, dh=dh, rtol=1e-10, atol=1e-10, max_it=10000
)
if not is_success:
    raise RuntimeError(f"PETSc solver failed: {msg_h}")
print(f"PETSc: {msg_h}")

# Face-centered velocity (MODFLOW-style: U at x-faces, V at y-faces)
U_face, V_face = hydraulic.compute_velocity(
    K, x_cc, y_cc, h, output_location="face"
)
# U_face (qxx), V_face (qyy) are (ny, nx) each — one face value per cell for Parcels grid
U = np.asarray(U_face, dtype=np.float32)
V = np.asarray(V_face, dtype=np.float32)

# Grid for Parcels: cell-centered (same shape as U, V)
x = np.linspace(0.5 * Lx / nx, Lx - 0.5 * Lx / nx, nx, dtype=np.float32)
y = np.linspace(0.5 * Ly / ny, Ly - 0.5 * Ly / ny, ny, dtype=np.float32)

data = {"U": U, "V": V}
dims = {"lon": x, "lat": y}

fieldset = parcels.FieldSet.from_data(
    data,
    dims,
    mesh="flat",
    allow_time_extrapolation=True,
)

# ----------------------------
# 2) Create particles at the left boundary
# ----------------------------
n_particles = 100
x_left = float(x[10])
y_release = np.linspace(float(y[10]), float(y[-10]), n_particles, dtype=np.float32)

pclass = parcels.JITParticle
pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=pclass,
    lon=np.full(n_particles, x_left),
    lat=y_release,
)

# ----------------------------
# 3) Execute (advect)
# ----------------------------
output_dt_sec = 0.2
dt_sec = 0.05

speed = np.sqrt(U * U + V * V)
vmax = float(np.nanmax(speed))
target_x = float(x[-10])
# Conservative runtime so at least one particle can reach right side.
runtime_sec = max(30.0, 5.0 * (target_x - x_left) / max(vmax, 1e-6))

output = pset.ParticleFile(name="traj.zarr", outputdt=timedelta(seconds=output_dt_sec))

pset.execute(
    parcels.AdvectionRK4,
    dt=timedelta(seconds=dt_sec),
    runtime=timedelta(seconds=runtime_sec),
    output_file=output,
)

print("Wrote traj.zarr")

import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_zarr("traj.zarr")

lon = ds["lon"].values
lat = ds["lat"].values

U_plot = fieldset.U.data[0]
V_plot = fieldset.V.data[0]
x_plot = fieldset.U.lon
y_plot = fieldset.U.lat

fig, ax = plt.subplots(figsize=(6, 6))

ax.streamplot(x_plot, y_plot, U_plot, V_plot, density=1.5)

def first_crossing_time(xp, dt_out, x_target):
    """Return first-passage time and index for x >= x_target, else (inf, None)."""
    valid = np.isfinite(xp)
    if not np.any(valid):
        return np.inf, None
    xvals = xp[valid]
    idx_valid = np.flatnonzero(valid)
    hit = np.where(xvals >= x_target)[0]
    if hit.size == 0:
        return np.inf, None
    k = int(hit[0])
    global_k = int(idx_valid[k])
    if k == 0:
        return global_k * dt_out, global_k
    x0 = xvals[k - 1]
    x1 = xvals[k]
    frac = 1.0 if abs(x1 - x0) < 1e-30 else float(np.clip((x_target - x0) / (x1 - x0), 0.0, 1.0))
    t_cross = (idx_valid[k - 1] + frac) * dt_out
    return float(t_cross), global_k

best_pid = None
best_time = np.inf
for p in range(lon.shape[1]):
    t_cross, _ = first_crossing_time(lon[:, p], output_dt_sec, target_x)
    if t_cross < best_time:
        best_time = t_cross
        best_pid = p

for p in range(lon.shape[1]):
    valid = np.isfinite(lon[:, p]) & np.isfinite(lat[:, p])
    if not np.any(valid):
        continue
    ax.plot(lon[valid, p], lat[valid, p], color="gray", lw=0.8, alpha=0.45)

if best_pid is not None and np.isfinite(best_time):
    valid = np.isfinite(lon[:, best_pid]) & np.isfinite(lat[:, best_pid])
    ax.plot(
        lon[valid, best_pid],
        lat[valid, best_pid],
        color="red",
        lw=2.2,
        label=f"Fastest path (t={best_time:.2f}s)",
    )
    print(f"Fastest particle: {best_pid}, first crossing time to x={target_x:.3f} is {best_time:.3f} s")
else:
    print("No particle reached the right boundary within runtime.")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Particle trajectories: fBm log-K → PETSc head → face-centered velocity")
ax.set_aspect("equal")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
if best_pid is not None and np.isfinite(best_time):
    ax.legend(loc="best")

out_fig = "particle_tracking_stream.png"
plt.savefig(out_fig, dpi=300)
print(f"Saved trajectory plot to {out_fig}")
plt.show()
