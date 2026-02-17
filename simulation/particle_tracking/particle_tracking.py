import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti


# Initialize Taichi on GPU
ti.init(arch=ti.gpu)


@dataclass(frozen=True)
class SimConfig:
    # Grid and particles
    grid_res: int = 512
    n_particles: int = 1_000_000

    # Time integration
    dt: float = 0.01
    max_steps: int = 5000

    # Domain and release
    source_x: float = 0.05
    target_x: float = 0.95
    y_min: float = 0.1
    y_max: float = 0.9

    # PAR2-like boundary behavior: if a candidate step exits domain in one
    # axis, keep the previous coordinate in that axis.
    closed_boundary: bool = True

    # If closed_boundary is False, particles exiting the domain get this time.
    exit_penalty_time: float = 1e9

    # Optional isotropic random walk term (0 disables diffusion).
    diffusion_sigma: float = 0.0


@ti.data_oriented
class ParticleTracking2D:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.L = cfg.grid_res
        self.N = cfg.n_particles
        self.dt = float(cfg.dt)
        self.max_steps = int(cfg.max_steps)
        self.source_x = float(cfg.source_x)
        self.target_x = float(cfg.target_x)
        self.y_min = float(cfg.y_min)
        self.y_max = float(cfg.y_max)
        self.closed_boundary = bool(cfg.closed_boundary)
        self.exit_penalty_time = float(cfg.exit_penalty_time)
        self.diffusion_sigma = float(cfg.diffusion_sigma)

        self.vel_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.L, self.L))
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.active = ti.field(dtype=ti.i32, shape=self.N)
        self.elapsed_time = ti.field(dtype=ti.f32, shape=self.N)
        self.final_times = ti.field(dtype=ti.f32, shape=self.N)
        self.active_count = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def _clamp01(self, x):
        return ti.min(1.0, ti.max(0.0, x))

    @ti.func
    def _gaussian(self):
        # Box-Muller transform from uniform random numbers.
        u1 = ti.max(ti.random(ti.f32), 1e-7)
        u2 = ti.random(ti.f32)
        r = ti.sqrt(-2.0 * ti.log(u1))
        theta = 2.0 * np.pi * u2
        return r * ti.cos(theta)

    @ti.func
    def bilinear_interp(self, p):
        """
        Robust bilinear interpolation.
        Coordinates are clamped first so weights always stay in [0, 1].
        """
        # Sample coordinates in index space [0, L-1]
        l_minus_1 = ti.cast(self.L - 1, ti.f32)
        u = self._clamp01(p.x) * l_minus_1
        v = self._clamp01(p.y) * l_minus_1

        x0 = ti.cast(ti.floor(u), ti.i32)
        y0 = ti.cast(ti.floor(v), ti.i32)
        x1 = ti.min(x0 + 1, self.L - 1)
        y1 = ti.min(y0 + 1, self.L - 1)

        wx = u - ti.cast(x0, ti.f32)
        wy = v - ti.cast(y0, ti.f32)

        v00 = self.vel_field[x0, y0]
        v10 = self.vel_field[x1, y0]
        v01 = self.vel_field[x0, y1]
        v11 = self.vel_field[x1, y1]
        return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

    @ti.kernel
    def init_velocity_field(self):
        """Synthetic velocity field with eddies and a slow central region."""
        for i, j in self.vel_field:
            # Normalized coordinates [0, 1]
            u = ti.cast(i, ti.f32) / ti.cast(self.L - 1, ti.f32)
            v = ti.cast(j, ti.f32) / ti.cast(self.L - 1, ti.f32)

            vel_x = 1.0
            vel_y = 0.0

            vel_x += 0.2 * ti.sin(2.0 * np.pi * v * 3.0)
            vel_y += 0.3 * ti.cos(2.0 * np.pi * u * 5.0)

            dist_center = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)
            if dist_center < 0.05:
                vel_x *= 0.1
                vel_y *= 0.1

            self.vel_field[i, j] = ti.Vector([vel_x, vel_y])

    @ti.kernel
    def init_particles(self):
        """Initialize particles uniformly along source line."""
        for i in range(self.N):
            frac = ti.cast(i, ti.f32) / ti.cast(self.N - 1, ti.f32)
            y_start = self.y_min + frac * (self.y_max - self.y_min)
            self.pos[i] = ti.Vector([self.source_x, y_start])
            self.active[i] = 1
            self.elapsed_time[i] = 0.0
            self.final_times[i] = ti.cast(self.exit_penalty_time, ti.f32)

    @ti.kernel
    def step_rk4(self, dt: ti.f32):
        """
        Advance all particles by one RK4 step.
        Fixes two issues in the original version:
        1) robust interpolation near/outside boundaries
        2) sub-step crossing time estimate for target plane
        """
        for i in self.pos:
            if self.active[i] == 1:
                p0 = self.pos[i]
                t0 = self.elapsed_time[i]

                # Deterministic RK4 drift
                k1 = self.bilinear_interp(p0)
                k2 = self.bilinear_interp(p0 + 0.5 * dt * k1)
                k3 = self.bilinear_interp(p0 + 0.5 * dt * k2)
                k4 = self.bilinear_interp(p0 + dt * k3)
                v_final = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                p1 = p0 + v_final * dt

                # Optional isotropic diffusion term
                if ti.static(self.diffusion_sigma > 0.0):
                    sigma_dt = self.diffusion_sigma * ti.sqrt(dt)
                    p1 += sigma_dt * ti.Vector([self._gaussian(), self._gaussian()])

                # PAR2-like closed boundary by default.
                if ti.static(self.closed_boundary):
                    if not (0.0 < p1.x < 1.0):
                        p1.x = p0.x
                    if not (0.0 < p1.y < 1.0):
                        p1.y = p0.y
                else:
                    if p1.x < 0.0 or p1.x > 1.0 or p1.y < 0.0 or p1.y > 1.0:
                        self.active[i] = 0
                        self.final_times[i] = ti.cast(self.exit_penalty_time, ti.f32)
                        continue

                t1 = t0 + dt

                # Sub-step crossing time to reduce dt quantization bias.
                if p0.x < self.target_x and p1.x >= self.target_x:
                    denom = p1.x - p0.x
                    frac = 1.0
                    if ti.abs(denom) > 1e-8:
                        frac = (self.target_x - p0.x) / denom
                        frac = ti.min(1.0, ti.max(0.0, frac))
                    self.final_times[i] = t0 + frac * dt
                    self.active[i] = 0
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                elif p1.x >= self.target_x:
                    self.final_times[i] = t1
                    self.active[i] = 0
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                else:
                    self.pos[i] = p1
                    self.elapsed_time[i] = t1
                    ti.atomic_add(self.active_count[None], 1)

    def run_pass1(self):
        print(f"Initializing {self.N} particles...")
        self.init_velocity_field()
        self.init_particles()

        print("Starting GPU simulation (Pass 1)...")
        start = time.time()
        steps_ran = 0

        for step in range(self.max_steps):
            self.active_count[None] = 0
            self.step_rk4(self.dt)
            steps_ran = step + 1
            if self.active_count[None] == 0:
                break

        ti.sync()
        elapsed = time.time() - start
        print(f"Simulation finished in {elapsed:.4f}s ({steps_ran} steps).")

        times_np = self.final_times.to_numpy()
        winner_idx = int(np.argmin(times_np))
        min_time = float(times_np[winner_idx])
        return winner_idx, min_time, times_np

    def _numpy_bilinear_interp(self, p, vel_np):
        l_minus_1 = float(self.L - 1)
        u = np.clip(float(p[0]), 0.0, 1.0) * l_minus_1
        v = np.clip(float(p[1]), 0.0, 1.0) * l_minus_1

        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1 = min(x0 + 1, self.L - 1)
        y1 = min(y0 + 1, self.L - 1)
        wx = u - x0
        wy = v - y0

        v00 = vel_np[x0, y0]
        v10 = vel_np[x1, y0]
        v01 = vel_np[x0, y1]
        v11 = vel_np[x1, y1]
        return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

    def replay_winner(self, winner_idx):
        """
        Re-run one particle on CPU to store full trajectory for plotting.
        For diffusion_sigma > 0, replay is deterministic and therefore only
        approximate (random increments are not replayed).
        """
        path_x = []
        path_y = []

        frac = winner_idx / (self.N - 1)
        y_start = self.y_min + frac * (self.y_max - self.y_min)
        p = np.array([self.source_x, y_start], dtype=np.float32)
        vel_np = self.vel_field.to_numpy()

        for _ in range(self.max_steps):
            path_x.append(float(p[0]))
            path_y.append(float(p[1]))

            if p[0] >= self.target_x:
                break

            k1 = self._numpy_bilinear_interp(p, vel_np)
            k2 = self._numpy_bilinear_interp(p + 0.5 * self.dt * k1, vel_np)
            k3 = self._numpy_bilinear_interp(p + 0.5 * self.dt * k2, vel_np)
            k4 = self._numpy_bilinear_interp(p + self.dt * k3, vel_np)
            v_final = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            p1 = p + v_final * self.dt

            if self.closed_boundary:
                if not (0.0 < p1[0] < 1.0):
                    p1[0] = p[0]
                if not (0.0 < p1[1] < 1.0):
                    p1[1] = p[1]

            p = p1

        return np.array(path_x), np.array(path_y)

    def plot_result(self, winner_idx, path_x, path_y):
        plt.figure(figsize=(10, 5))

        vf = self.vel_field.to_numpy()
        y_grid, x_grid = np.mgrid[0:self.L, 0:self.L]
        q_step = 16
        plt.quiver(
            x_grid[::q_step, ::q_step] / (self.L - 1),
            y_grid[::q_step, ::q_step] / (self.L - 1),
            vf[::q_step, ::q_step, 0],
            vf[::q_step, ::q_step, 1],
            color="lightgray",
            scale=30,
        )

        plt.plot(path_x, path_y, "r-", linewidth=2, label="Fastest path")
        plt.axvline(self.source_x, color="g", linestyle="--", label="Source")
        plt.axvline(self.target_x, color="b", linestyle="--", label="Target")

        plt.title(f"Fastest Path Optimization (N={self.N}, winner={winner_idx})")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()


def main():
    cfg = SimConfig()
    sim = ParticleTracking2D(cfg)

    winner_idx, min_time, _ = sim.run_pass1()
    if min_time >= cfg.exit_penalty_time * 0.9:
        print("No particles reached the target!")
        return

    print(f"Fastest particle index: {winner_idx}")
    print(f"Fastest time: {min_time:.6f}")

    print("Replaying winner trajectory...")
    path_x, path_y = sim.replay_winner(winner_idx)
    sim.plot_result(winner_idx, path_x, path_y)


if __name__ == "__main__":
    main()
    