"""
Parcels JIT needs a working C compiler. If you see "cannot execute 'cc1'" from
conda's compiler, use the system compiler instead before running:
    export CC=gcc
    export CXX=g++
    python test.py
"""
import numpy as np
from datetime import timedelta
import parcels

# ----------------------------
# 1) Create a static velocity field U(x,y), V(x,y)
# ----------------------------
Lx, Ly = 1.0, 1.0
nx, ny = 101, 101
x = np.linspace(0, Lx, nx, dtype=np.float32)
y = np.linspace(0, Ly, ny, dtype=np.float32)

X, Y = np.meshgrid(x, y, indexing="xy")

# Left-to-right velocity field: uniform flow with optional slight variation
# U > 0 (flow to the right), V ≈ 0
U0 = 1.0
U = np.full_like(X, U0, dtype=np.float32)
V = np.zeros_like(X, dtype=np.float32)

data = {"U": U, "V": V}
dims = {"lon": x, "lat": y}    # yes, Parcels uses names lon/lat even for flat meshes

fieldset = parcels.FieldSet.from_data(
    data,
    dims,
    mesh="flat",  # important: "flat" => no spherical metrics
    allow_time_extrapolation=True  # harmless here; helpful if a time axis exists
)

# ----------------------------
# 2) Create particles at the left boundary
# ----------------------------
n_particles = 11
x_left = 0.02  # slightly inside the domain
y_release = np.linspace(0.1, 0.9, n_particles, dtype=np.float32)

pclass = parcels.JITParticle  # fast; use ScipyParticle for easier debugging
pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=pclass,
    lon=np.full(n_particles, x_left),
    lat=y_release,
)

# ----------------------------
# 3) Execute (advect)
# ----------------------------
# With U=1, domain [0,1]: particle at x_left=0.02 reaches x=1 at t=(1-0.02)/1=0.98 s.
# Keep runtime < 0.98 s so no particle samples out-of-bound.
output = pset.ParticleFile(name="traj.zarr", outputdt=timedelta(seconds=0.02))

pset.execute(
    parcels.AdvectionRK4,
    dt=timedelta(seconds=0.01),
    runtime=timedelta(seconds=0.95),
    output_file=output,
)

# output.close()
print("Wrote traj.zarr")

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ds = xr.open_zarr("traj.zarr")

lon = ds["lon"].values  # shape (time, particle)
lat = ds["lat"].values

U = fieldset.U.data[0]
V = fieldset.V.data[0]
x = fieldset.U.lon
y = fieldset.U.lat

fig, ax = plt.subplots(figsize=(6, 6))

ax.streamplot(x, y, U, V, density=1.5)

for p in range(lon.shape[1]):
    ax.plot(lon[:, p], lat[:, p], color="red")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Particle trajectories: left-to-right flow (released at left boundary)")
ax.set_aspect("equal")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

out_fig = "particle_tracking_stream.png"
plt.savefig(out_fig, dpi=300)
print(f"Saved trajectory plot to {out_fig}")
plt.show()