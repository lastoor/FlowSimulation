"""
CPU particle tracking using RK45 (Dormand–Prince) integration with Numba.
"""

import math
import numpy as np
from numba import njit


@njit(inline='always')
def find_le_index(arr, val):
    n = arr.size
    if val <= arr[0]:
        return 0
    if val >= arr[n-2]:
        return n-2
    lo, hi = 0, n-1
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo


@njit(inline='always')
def bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y):
    if x <= xgrid[0]:
        ix = 0
    elif x >= xgrid[nx-1]:
        ix = nx - 2
    else:
        ix = find_le_index(xgrid, x)

    if y <= ygrid[0]:
        iy = 0
    elif y >= ygrid[ny-1]:
        iy = ny - 2
    else:
        iy = find_le_index(ygrid, y)

    x0, x1 = xgrid[ix], xgrid[ix+1]
    y0, y1 = ygrid[iy], ygrid[iy+1]

    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)

    i00 = iy*nx + ix
    i10 = iy*nx + ix + 1
    i01 = (iy+1)*nx + ix
    i11 = (iy+1)*nx + ix + 1

    vx0 = vx[i00] + tx*(vx[i10] - vx[i00])
    vx1 = vx[i01] + tx*(vx[i11] - vx[i01])
    vy0 = vy[i00] + tx*(vy[i10] - vy[i00])
    vy1 = vy[i01] + tx*(vy[i11] - vy[i01])

    return vx0 + ty*(vx1 - vx0), vy0 + ty*(vy1 - vy0)


@njit
def norm2(v0, v1):
    """Simple L2 norm function."""
    return math.sqrt(v0**2 + v1**2)


@njit
def rhs_arclength(x, y, xgrid, ygrid, vx, vy, nx, ny, eps_speed):
    """
    For z=[x,y,t], dz/ds = [vx/|v|, vy/|v|, 1/|v|].
    Runs on a single CPU thread.
    """
    vx_at_p, vy_at_p = bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y)
    sp = norm2(vx_at_p, vy_at_p)

    if not math.isfinite(sp) or sp < eps_speed:
        return 0.0, 0.0, 1e30, sp

    inv = 1.0 / sp
    rate_x = vx_at_p * inv
    rate_y = vy_at_p * inv
    rate_t = inv

    return rate_x, rate_y, rate_t, sp


@njit
def rk45_one_particle(xgrid, ygrid,
                      vx, vy, nx, ny,
                      x0, y0,
                      right_x, xmin, xmax, ymin, ymax,
                      eps_speed, ds_min, ds_max, tol,
                      y_policy,
                      max_points, max_iters,
                      A, B5, B4):
    """
    Returns: x_path, y_path, t_path, length, status
    """
    x = x0
    y = y0
    t = 0.0
    ds = ds_max * 0.25
    sp = 0.0
    st = 4  # default fail

    x_path = np.zeros(max_points, np.float32)
    y_path = np.zeros(max_points, np.float32)
    t_path = np.zeros(max_points, np.float32)

    x_path[0] = x
    y_path[0] = y
    t_path[0] = t
    count = 1

    for it in range(max_iters):
        if count >= max_points:
            break

        # Boundary conditions
        if x >= right_x:
            st = 0
            break
        if x < xmin or x > xmax:
            st = 2
            break

        if y < ymin or y > ymax:
            if y_policy == 0:   # stop
                st = 3
                break
            elif y_policy == 1:  # reflect
                span = ymax - ymin
                Y = ymin + (y-ymin) % (2*span)
                if Y < ymin: Y += 2*span
                if Y > ymax: Y = ymax - (Y-ymax)
                y = Y
            elif y_policy == 2:  # clip
                y = min(max(y, ymin), ymax)

        # Guard step
        ds = min(max(ds, ds_min), ds_max)

        # RK45 stage buffers
        kx = np.zeros(7, np.float32)
        ky = np.zeros(7, np.float32)
        kt = np.zeros(7, np.float32)

        # Stage k1
        kx[0], ky[0], kt[0], sp = rhs_arclength(x, y, xgrid, ygrid,
                                                vx, vy, nx, ny, eps_speed)
        if sp < eps_speed or not math.isfinite(sp):
            st = 1
            break

        # Stages k2..k6
        for s in range(1, 6):
            xi, yi, ti = x, y, t
            for j in range(s):
                xi += ds * A[s,j] * kx[j]
                yi += ds * A[s,j] * ky[j]
                ti += ds * A[s,j] * kt[j]

            kx[s], ky[s], kt[s], _ = rhs_arclength(xi, yi, xgrid, ygrid,
                                                   vx, vy, nx, ny, eps_speed)

        # 5th order solution
        x5, y5, t5 = x, y, t
        for j in range(6):
            x5 += ds * B5[j] * kx[j]
            y5 += ds * B5[j] * ky[j]
            t5 += ds * B5[j] * kt[j]

        # 4th order estimate
        x4, y4, t4 = x, y, t
        for j in range(6):
            x4 += ds * B4[j] * kx[j]
            y4 += ds * B4[j] * ky[j]
            t4 += ds * B4[j] * kt[j]
        x4 += ds * B4[6] * kx[0]
        y4 += ds * B4[6] * ky[0]
        t4 += ds * B4[6] * kt[0]

        # error
        err = math.sqrt((x5-x4)**2 + (y5-y4)**2) + 1e-30

        if err <= tol:  # accept step
            x, y, t = x5, y5, t5

            x_path[count] = x
            y_path[count] = y
            t_path[count] = t
            count += 1

            ds = ds * min(5.0, max(0.2, 0.9 * (tol/err)**0.2))
        else:           # reject
            ds = ds * max(0.1, 0.5 * (tol/err)**0.25)

        # Detect crossing
        if x < right_x <= x5:
            tau = (right_x - x) / ((x5-x) + 1e-30)
            tau = min(max(tau, 0), 1)
            x = right_x
            y = y + tau * (y5-y)
            t = t + tau * (t5-t)

            x_path[count] = x
            y_path[count] = y
            t_path[count] = t
            count += 1
            st = 0
            break

    return x_path, y_path, t_path, count, st


def particle_tracking_cpu(params):
    x = params["x"].astype(np.float32)
    y = params["y"].astype(np.float32)
    vx = params["vx"].astype(np.float32)
    vy = params["vy"].astype(np.float32)
    seeds_x = params["seeds_x"].astype(np.float32)
    seeds_y = params["seeds_y"].astype(np.float32)

    nx = vx.shape[1]
    ny = vx.shape[0]
    vx_flat = vx.ravel()
    vy_flat = vy.ravel()

    right_x = params.get("right_x", x[-1])
    eps_speed = params.get("eps_speed", 1e-16)
    ds_min = params.get("ds_min", 1e-5)
    ds_max = params.get("ds_max", 1.0)
    tol = params.get("tol", 1e-4)
    y_policy = {"stop":0, "reflect":1, "clip":2}[params.get("y_policy", "stop")]
    max_points = params.get("max_points", 2000)
    max_iters = params.get("max_iters", 200000)

    # RK45 coefficients
    A = np.zeros((6,6),np.float32)
    A[1,0] = 1/5
    A[2,0:2] = [3/40, 9/40]
    A[3,0:3] = [44/45,-56/15,32/9]
    A[4,0:4] = [19372/6561,-25360/2187,64448/6561,-212/729]
    A[5,0:5] = [9017/3168,-355/33,46732/5247,49/176,-5103/18656]

    B5 = np.array([35/384, 0., 500/1113, 125/192,-2187/6784,11/84], np.float32)
    B4 = np.array([5179/57600,0.,7571/16695,393/640,-92097/339200,187/2100,1/40], np.float32)

    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]

    nP = seeds_x.size

    paths_x = np.zeros((nP, max_points), np.float32)
    paths_y = np.zeros((nP, max_points), np.float32)
    paths_t = np.zeros((nP, max_points), np.float32)
    lengths = np.zeros(nP, np.int32)
    status  = np.zeros(nP, np.int32)

    for i in range(nP):
        xp, yp, tp, L, st = rk45_one_particle(
            x, y, vx_flat, vy_flat, nx, ny,
            seeds_x[i], seeds_y[i],
            right_x, xmin, xmax, ymin, ymax,
            eps_speed, ds_min, ds_max, tol,
            y_policy,
            max_points, max_iters,
            A, B5, B4
        )
        paths_x[i,:] = xp
        paths_y[i,:] = yp
        paths_t[i,:] = tp
        lengths[i] = L
        status[i] = st

    return paths_x, paths_y, paths_t, lengths, status


