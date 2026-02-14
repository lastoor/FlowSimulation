"""
Hydraulic head solving and velocity computation.
"""

import numpy as np


def solve_head(K, x, y, dh=0.01, rtol=1e-6, atol=1e-20, max_it=10000):
    """
    Solve ∇·(K∇h) = 0 in a 2D domain with:
    - Dirichlet BCs on left/right: h(left)=dh*x_len, h(right)=0
    - No-flow BCs on top/bottom.

    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx).
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions.
    dh : float
        Mean hydraulic head gradient between left and right boundaries.

    Returns
    -------
    h : 2D np.ndarray
        Computed hydraulic head field.
    msg : str
        PETSc convergence message.
    is_success : bool
        Whether the solver converged.
    """
    # Import PETSc inside function to avoid top-level dependency
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc

    ny, nx = K.shape
    dx = np.diff(np.concatenate(([0], x + 0.5*(x[1]-x[0]))))
    dy = np.diff(np.concatenate(([0], y + 0.5*(y[1]-y[0]))))
    dz = 1.0

    def lid(i, j): return i*nx + j
    def hmean(a, b): return 2*a*b/(a+b + 1e-300)

    dx_face = np.empty(nx+1)
    dx_face[1:-1] = 0.5*(dx[:-1] + dx[1:])
    dx_face[0] = dx[0]
    dx_face[-1] = dx[-1]

    dy_face = np.empty(ny+1)
    dy_face[1:-1] = 0.5*(dy[:-1] + dy[1:])
    dy_face[0] = dy[0]
    dy_face[-1] = dy[-1]

    # PETSc assembly
    N = nx * ny
    A = PETSc.Mat().create()
    A.setSizes((N, N))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setPreallocationNNZ(5)

    b = PETSc.Vec().createMPI(N)
    b.set(0.0)

    LEFT_VAL, RIGHT_VAL = dh*np.abs(x[-1]-x[0]), 0.0

    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = divmod(row, nx)
        is_left = (j == 0)
        is_right = (j == nx-1)

        if is_left or is_right:
            A.setValue(row, row, 1.0)
            b.setValue(row, LEFT_VAL if is_left else RIGHT_VAL)
            continue

        Ax = dy[i]*dz
        Ay = dx[j]*dz

        KW = hmean(K[i, j], K[i, j-1])
        GW = KW * Ax / dx_face[j]

        KE = hmean(K[i, j], K[i, j+1])
        GE = KE * Ax / dx_face[j+1]

        GS = hmean(K[i, j], K[i-1, j]) * Ay / dy_face[i] if i > 0 else 0.0
        GN = hmean(K[i, j], K[i+1, j]) * Ay / dy_face[i+1] if i < ny-1 else 0.0

        diag = GE + GW + GN + GS
        A.setValue(row, row, diag)

        if j-1 == 0:
            b.setValue(row, GW*LEFT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j-1), -GW)

        if j+1 == nx-1:
            b.setValue(row, GE*RIGHT_VAL, addv=True)
        else:
            A.setValue(row, lid(i, j+1), -GE)

        if i > 0:
            A.setValue(row, lid(i-1, j), -GS)
        if i < ny-1:
            A.setValue(row, lid(i+1, j), -GN)

    A.assemblyBegin(); A.assemblyEnd()
    b.assemblyBegin(); b.assemblyEnd()

    x_vec = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    try:
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
    except PETSc.Error:
        pc.setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    ksp.setFromOptions()
    ksp.solve(b, x_vec)

    h_flat = x_vec.getArray()
    h = h_flat.reshape((ny, nx))

    msg = (f"ConvergedReason: {ksp.getConvergedReason()}, "
           f"Iterations: {ksp.getIterationNumber()}, "
           f"Residual: {ksp.getResidualNorm():.3e}")
    is_success = True if (ksp.getConvergedReason() > 0) else False
    return h, msg, is_success


def compute_velocity(K, x, y, h):
    """
    Compute Darcy velocity components (vx, vy) from head field h(x,y).

    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx)
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions
    h : 2D np.ndarray
        Hydraulic head field, shape (ny, nx)

    Returns
    -------
    vx, vy : 2D np.ndarray
        Darcy velocity components in x and y directions
        vx = -K * ∂h/∂x, vy = -K * ∂h/∂y
    """
    ny, nx = h.shape
    dhdx = np.zeros_like(h)
    dhdy = np.zeros_like(h)

    # --- x-derivative (central interior, one-sided edges)
    for i in range(ny):
        dhdx[i, 1:-1] = (h[i, 2:] - h[i, :-2]) / (x[2:] - x[:-2])
        dhdx[i, 0] = (h[i, 1] - h[i, 0]) / (x[1] - x[0])
        dhdx[i, -1] = (h[i, -1] - h[i, -2]) / (x[-1] - x[-2])

    # --- y-derivative (central interior, one-sided edges)
    for j in range(nx):
        dhdy[1:-1, j] = (h[2:, j] - h[:-2, j]) / (y[2:] - y[:-2])
        dhdy[0, j] = (h[1, j] - h[0, j]) / (y[1] - y[0])
        dhdy[-1, j] = (h[-1, j] - h[-2, j]) / (y[-1] - y[-2])

    # --- Darcy velocity
    vx = -K * dhdx
    vy = -K * dhdy
    return vx, vy