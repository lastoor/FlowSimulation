"""
GPU particle tracking using RK45 (Dormand–Prince) with Numba CUDA.
"""

import math
import numpy as np
from numba import cuda, float32


def _build_faces_from_centers(centers):
    centers = np.asarray(centers, dtype=np.float32).ravel()
    if centers.size < 2:
        raise ValueError("center coordinates must contain at least 2 values.")
    faces = np.empty(centers.size + 1, dtype=np.float32)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    faces[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    faces[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return faces

# =========================
# Device-side helpers
# =========================


@cuda.jit(device=True, inline=True)
def find_le_index(arr, n, val):
    """Largest idx i s.t. arr[i] <= val < arr[i+1]; clamp to [0, n-2]."""
    lo, hi = 0, n - 1
    if val <= arr[0]:
        return 0
    if val >= arr[n - 2]:
        return n - 2
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    return lo


@cuda.jit(device=True, inline=True)
def bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y, out_v):
    """out_v[0]=vx(x,y), out_v[1]=vy(x,y), bilinear on rect grid (x fastest)."""
    if x <= xgrid[0]:
        ix = 0
    elif x >= xgrid[nx - 1]:
        ix = nx - 2
    else:
        ix = find_le_index(xgrid, nx, x)

    if y <= ygrid[0]:
        iy = 0
    elif y >= ygrid[ny - 1]:
        iy = ny - 2
    else:
        iy = find_le_index(ygrid, ny, y)

    x0, x1 = xgrid[ix], xgrid[ix + 1]
    y0, y1 = ygrid[iy], ygrid[iy + 1]
    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)

    i00 = iy * nx + ix
    i10 = iy * nx + (ix + 1)
    i01 = (iy + 1) * nx + ix
    i11 = (iy + 1) * nx + (ix + 1)

    vx00, vx10 = vx[i00], vx[i10]
    vx01, vx11 = vx[i01], vx[i11]
    vy00, vy10 = vy[i00], vy[i10]
    vy01, vy11 = vy[i01], vy[i11]

    vx0 = vx00 + tx * (vx10 - vx00)
    vx1 = vx01 + tx * (vx11 - vx01)
    vy0 = vy00 + tx * (vy10 - vy00)
    vy1 = vy01 + tx * (vy11 - vy01)
    out_v[0] = vx0 + ty * (vx1 - vx0)
    out_v[1] = vy0 + ty * (vy1 - vy0)


@cuda.jit(device=True, inline=True)
def _center_cell_index(grid, n, val):
    if val <= grid[0]:
        return 0
    if val >= grid[n - 1]:
        return n - 1
    return find_le_index(grid, n, val)


@cuda.jit(device=True, inline=True)
def face_sample(xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, x, y, out_v):
    iy = _center_cell_index(ygrid, ny, y)
    ix_cell = _center_cell_index(xgrid, nx, x)

    if x <= x_faces[0]:
        ix = 0
    elif x >= x_faces[nx]:
        ix = nx - 1
    else:
        ix = find_le_index(x_faces, nx + 1, x)

    if ix <= 0:
        q_xL = vx[iy * nx]
        q_xR = q_xL
        x_L, x_R = x_faces[0], x_faces[1]
    else:
        q_xL = vx[iy * nx + (ix - 1)]
        q_xR = vx[iy * nx + ix]
        x_L, x_R = x_faces[ix], x_faces[ix + 1]
    dx_seg = x_R - x_L + 1e-30
    vx_val = q_xL + (x - x_L) / dx_seg * (q_xR - q_xL)

    if y <= y_faces[0]:
        iy_seg = 0
    elif y >= y_faces[ny]:
        iy_seg = ny - 1
    else:
        iy_seg = find_le_index(y_faces, ny + 1, y)

    if iy_seg <= 0:
        q_yB = vy[ix_cell]
        q_yT = q_yB
        y_B, y_T = y_faces[0], y_faces[1]
    else:
        q_yB = vy[(iy_seg - 1) * nx + ix_cell]
        q_yT = vy[iy_seg * nx + ix_cell]
        y_B, y_T = y_faces[iy_seg], y_faces[iy_seg + 1]
    dy_seg = y_T - y_B + 1e-30
    vy_val = q_yB + (y - y_B) / dy_seg * (q_yT - q_yB)

    out_v[0] = vx_val
    out_v[1] = vy_val


@cuda.jit(device=True, inline=True)
def norm2(x, y):
    return math.sqrt(x * x + y * y)


@cuda.jit(device=True)
def rhs_arclength(
    x, y, xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, velocity_mode, eps_speed, out_rhs, out_speed
):
    """For z=[x,y,t], dz/ds = [vx/|v|, vy/|v|, 1/|v|]."""
    v = cuda.local.array(2, dtype=float32)
    if velocity_mode == 0:
        bilinear_sample(xgrid, ygrid, vx, vy, nx, ny, x, y, v)
    else:
        face_sample(xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, x, y, v)
    sp = norm2(v[0], v[1])
    if not math.isfinite(sp) or sp < eps_speed:
        out_rhs[0] = 0.0
        out_rhs[1] = 0.0
        out_rhs[2] = 1e30
        out_speed[0] = sp
        return
    inv = 1.0 / sp
    out_rhs[0] = v[0] * inv
    out_rhs[1] = v[1] * inv
    out_rhs[2] = inv
    out_speed[0] = sp


# Dormand–Prince RK45 coefficients (float32), uploaded once to device
_A = np.zeros((6, 6), dtype=np.float32)
_A[1, 0] = 1/5
_A[2, 0:2] = [3/40, 9/40]
_A[3, 0:3] = [44/45, -56/15, 32/9]
_A[4, 0:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
_A[5, 0:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]

_B5 = np.array([35/384, 0., 500/1113, 125/192, -2187/6784, 11/84], dtype=np.float32)
_B4 = np.array([5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float32)

# Lazy device upload (avoids CUDA init when no GPU, e.g. in tests)
_A_d = _B5_d = _B4_d = None


def _get_rk45_device_arrays():
    """Upload RK45 coefficient arrays to device on first use."""
    global _A_d, _B5_d, _B4_d
    if _A_d is None:
        _A_d = cuda.to_device(_A)
        _B5_d = cuda.to_device(_B5)
        _B4_d = cuda.to_device(_B4)
    return _A_d, _B5_d, _B4_d


@cuda.jit
def rk45_arclength_kernel(
    xgrid, ygrid, vx, vy, nx, ny,
    x_faces, y_faces, velocity_mode,
    seeds_x, seeds_y,
    right_x, xmin, xmax, ymin, ymax,
    eps_speed, ds_min, ds_max, tol,
    y_policy, max_points, max_iters,
    A_tbl, B5_tbl, B4_tbl,
    paths_x, paths_y, paths_t, lengths, status,
    info_code, info_value, info_speed,
):
    """
    One thread per particle. status: 0=ok/hit_right, 1=stagnation,
    2=left_domain_x, 3=left_domain_y, 4=fail.
    """
    pid = cuda.grid(1)
    nP = seeds_x.size
    if pid >= nP:
        return

    x = float(seeds_x[pid])
    y = float(seeds_y[pid])
    t = 0.0
    sp = 0.0
    err = 0.0

    count = 0
    if count < max_points:
        paths_x[pid, count] = x
        paths_y[pid, count] = y
        paths_t[pid, count] = t
        count += 1

    ds = ds_max * 0.25
    iters = 0
    st = 4

    kx = cuda.local.array(7, dtype=float32)
    ky = cuda.local.array(7, dtype=float32)
    kt = cuda.local.array(7, dtype=float32)
    rhs = cuda.local.array(3, dtype=float32)
    spv = cuda.local.array(1, dtype=float32)

    while iters < max_iters and count < max_points:
        iters += 1

        if x >= right_x:
            st = 0
            break
        if x <= xmin or x >= xmax:
            st = 2
            break
        if y < ymin or y > ymax:
            if y_policy == 0:
                st = 3
                break
            elif y_policy == 1:
                span = ymax - ymin
                Y = ymin + math.fmod(y - ymin, 2.0 * span)
                if Y < ymin:
                    Y += 2.0 * span
                if Y > ymax:
                    Y = ymax - (Y - ymax)
                y = Y
            elif y_policy == 2:
                if y < ymin:
                    y = ymin
                if y > ymax:
                    y = ymax

        if ds < ds_min:
            ds = ds_min
        if ds > ds_max:
            ds = ds_max

        rhs_arclength(
            x, y, xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, velocity_mode, eps_speed, rhs, spv
        )
        sp = spv[0]
        if not math.isfinite(sp) or sp < eps_speed:
            st = 1
            break
        kx[0], ky[0], kt[0] = rhs[0], rhs[1], rhs[2]

        for s in range(1, 6):
            xi, yi, ti = x, y, t
            for j in range(s):
                a = A_tbl[s, j]
                xi += ds * a * kx[j]
                yi += ds * a * ky[j]
                ti += ds * a * kt[j]
            rhs_arclength(
                xi, yi, xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, velocity_mode, eps_speed, rhs, spv
            )
            kx[s], ky[s], kt[s] = rhs[0], rhs[1], rhs[2]

        x5, y5, t5 = x, y, t
        for j in range(6):
            x5 += ds * B5_tbl[j] * kx[j]
            y5 += ds * B5_tbl[j] * ky[j]
            t5 += ds * B5_tbl[j] * kt[j]

        rhs_arclength(
            x5, y5, xgrid, ygrid, x_faces, y_faces, vx, vy, nx, ny, velocity_mode, eps_speed, rhs, spv
        )
        kx[6], ky[6], kt[6] = rhs[0], rhs[1], rhs[2]

        x4, y4, t4 = x, y, t
        for j in range(7):
            x4 += ds * B4_tbl[j] * kx[j]
            y4 += ds * B4_tbl[j] * ky[j]
            t4 += ds * B4_tbl[j] * kt[j]

        ex = x5 - x4
        ey = y5 - y4
        err = math.sqrt(ex * ex + ey * ey) + 1e-30

        if err <= tol:
            x_prev, y_prev, t_prev = x, y, t
            if (x_prev < right_x) and (x5 >= right_x):
                tau = (right_x - x_prev) / ((x5 - x_prev) + 1e-30)
                if tau < 0.0:
                    tau = 0.0
                if tau > 1.0:
                    tau = 1.0
                x = right_x
                y = y_prev + tau * (y5 - y_prev)
                t = t_prev + tau * (t5 - t_prev)
                if count < max_points:
                    paths_x[pid, count] = x
                    paths_y[pid, count] = y
                    paths_t[pid, count] = t
                    count += 1
                st = 0
                break

            x, y, t = x5, y5, t5
            if count < max_points:
                paths_x[pid, count] = x
                paths_y[pid, count] = y
                paths_t[pid, count] = t
                count += 1
            ds = ds * min(5.0, max(0.2, 0.9 * math.pow(tol / err, 0.2)))
        else:
            ds = ds * max(0.1, 0.5 * math.pow(tol / err, 0.25))

    info_c = -1
    info_v = 0.0
    info_s = sp

    if ds < ds_min:
        info_c, info_v = 4, ds
    elif ds > ds_max:
        info_c, info_v = 5, ds
    elif err > tol:
        info_c, info_v = 6, err
    elif count >= max_points:
        info_c, info_v = 7, float(count)
    elif iters >= max_iters:
        info_c, info_v = 8, float(iters)
    elif not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(t):
        info_c, info_v = 9, 0.0

    info_code[pid] = info_c
    info_value[pid] = info_v
    info_speed[pid] = info_s

    lengths[pid] = count
    status[pid] = st if (st != 4 or iters < max_iters) else 4


def gpu_track_particles_arclength(
    xgrid, ygrid, vx, vy,
    seeds_x, seeds_y,
    right_x=None,
    eps_speed=1e-16, ds_min=1e-5, ds_max=1.0, tol=1e-4,
    y_policy="stop",
    max_points=2000, max_iters=200000,
    threads_per_block=256, stream=None, dtype=np.float32,
    velocity_location="cell",
    x_cell_centers=None, y_cell_centers=None,
):
    """
    Run RK45 particle tracking on GPU. Returns (paths_x, paths_y, paths_t,
    lengths, status, info_code, info_value, info_speed).
    """
    xgrid = np.asarray(xgrid, dtype=dtype)
    ygrid = np.asarray(ygrid, dtype=dtype)
    vx = np.asarray(vx, dtype=dtype, order="C")
    vy = np.asarray(vy, dtype=dtype, order="C")
    seeds_x = np.asarray(seeds_x, dtype=dtype)
    seeds_y = np.asarray(seeds_y, dtype=dtype)

    ny, nx = vx.shape
    if xgrid.size != nx or ygrid.size != ny:
        raise ValueError("xgrid/ygrid lengths must match vx/vy shape.")
    if velocity_location not in ("cell", "face"):
        raise ValueError("velocity_location must be 'cell' or 'face'.")
    velocity_mode = np.int32(0 if velocity_location == "cell" else 1)
    if velocity_mode == 1:
        x_centers = np.asarray(x_cell_centers if x_cell_centers is not None else xgrid, dtype=dtype).ravel()
        y_centers = np.asarray(y_cell_centers if y_cell_centers is not None else ygrid, dtype=dtype).ravel()
        if x_centers.size != nx or y_centers.size != ny:
            raise ValueError("x_cell_centers length must be nx; y_cell_centers length must be ny.")
    else:
        x_centers = xgrid
        y_centers = ygrid
    x_faces = _build_faces_from_centers(x_centers).astype(dtype, copy=False)
    y_faces = _build_faces_from_centers(y_centers).astype(dtype, copy=False)

    xmin, xmax = float(xgrid.min()), float(xgrid.max())
    ymin, ymax = float(ygrid.min()), float(ygrid.max())
    if right_x is None:
        right_x = xmax

    d_xgrid = cuda.to_device(xgrid)
    d_ygrid = cuda.to_device(ygrid)
    d_vx = cuda.to_device(vx.ravel())
    d_vy = cuda.to_device(vy.ravel())
    d_x_faces = cuda.to_device(x_faces)
    d_y_faces = cuda.to_device(y_faces)
    d_seeds_x = cuda.to_device(seeds_x)
    d_seeds_y = cuda.to_device(seeds_y)

    nP = seeds_x.size
    paths_x = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    paths_y = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    paths_t = cuda.device_array((nP, max_points), dtype=dtype, stream=stream)
    lengths = cuda.device_array(nP, dtype=np.int32, stream=stream)
    status = cuda.device_array(nP, dtype=np.int32, stream=stream)
    info_code = cuda.device_array(nP, dtype=np.int32, stream=stream)
    info_value = cuda.device_array(nP, dtype=dtype, stream=stream)
    info_speed = cuda.device_array(nP, dtype=dtype, stream=stream)

    ypol = {"stop": 0, "reflect": 1, "clip": 2}[y_policy]
    blocks = (nP + threads_per_block - 1) // threads_per_block
    A_d, B5_d, B4_d = _get_rk45_device_arrays()

    rk45_arclength_kernel[blocks, threads_per_block, stream](
        d_xgrid, d_ygrid,
        d_vx, d_vy, np.int32(nx), np.int32(ny),
        d_x_faces, d_y_faces, velocity_mode,
        d_seeds_x, d_seeds_y,
        dtype(right_x), dtype(xmin), dtype(xmax), dtype(ymin), dtype(ymax),
        dtype(eps_speed), dtype(ds_min), dtype(ds_max), dtype(tol),
        np.int32(ypol), np.int32(max_points), np.int32(max_iters),
        A_d, B5_d, B4_d,
        paths_x, paths_y, paths_t, lengths, status,
        info_code, info_value, info_speed,
    )
    cuda.synchronize()

    return (
        paths_x.copy_to_host(), paths_y.copy_to_host(), paths_t.copy_to_host(),
        lengths.copy_to_host(), status.copy_to_host(),
        info_code.copy_to_host(), info_value.copy_to_host(), info_speed.copy_to_host(),
    )
