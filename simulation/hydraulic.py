"""
Hydraulic head solving and velocity computation.
"""

import numpy as np


def solve_head(K, x, y, dh=0.01, rtol=1e-10, atol=1e-10, max_it=10000):
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
    # dx = np.diff(np.concatenate(([0], x + 0.5*(x[1]-x[0]))))
    # dy = np.diff(np.concatenate(([0], y + 0.5*(y[1]-y[0]))))
    dx = np.full(nx, 1.0/nx)
    dy = np.full(ny, 1.0/ny)
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


def compute_velocity(K, x, y, h, output_location="cell"):
    """
    Compute Darcy velocity using MODFLOW-consistent face discretization.
    Kept for reference; compute_velocity implements the same formulation per USGS TM6-A16 Ch. 2.

    Parameters
    ----------
    K : 2D np.ndarray
        Hydraulic conductivity at cell centers, shape (ny, nx).
    x, y : 1D np.ndarray
        Cell-center coordinates in x and y directions.
    h : 2D np.ndarray
        Hydraulic head field at cell centers, shape (ny, nx).
    output_location : str, optional
        Output velocity location:
        - "cell": return cell-centered velocity fields (ny, nx) via averaging.
        - "face": return face-centered velocity fields:
          vx -> (ny, nx+1), vy -> (ny+1, nx).
        Default is "cell".

    Returns
    -------
    vx, vy : 2D np.ndarray
        Darcy velocity components.
    """
    if output_location not in ("cell", "face"):
        raise ValueError("output_location must be either 'cell' or 'face'")

    ny, nx = h.shape

    def hmean(a, b):
        return 2.0 * a * b / (a + b + 1e-300)

    # Build cell widths from center coordinates using midpoint edges.
    # This avoids spuriously halving the first cell when coordinates start at 0.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != nx or y.size != ny:
        raise ValueError("x and y must match h dimensions")
    if nx < 2 or ny < 2:
        raise ValueError("compute_velocity requires at least 2 cells in each direction")

    x_edges = np.empty(nx + 1, dtype=float)
    x_edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    x_edges[0] = x[0] - 0.5 * (x[1] - x[0])
    x_edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    dx = np.diff(x_edges)

    y_edges = np.empty(ny + 1, dtype=float)
    y_edges[1:-1] = 0.5 * (y[:-1] + y[1:])
    y_edges[0] = y[0] - 0.5 * (y[1] - y[0])
    y_edges[-1] = y[-1] + 0.5 * (y[-1] - y[-2])
    dy = np.diff(y_edges)

    dx_face = np.empty(nx + 1)
    dx_face[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    dx_face[0] = dx[0]
    dx_face[-1] = dx[-1]

    dy_face = np.empty(ny + 1)
    dy_face[1:-1] = 0.5 * (dy[:-1] + dy[1:])
    dy_face[0] = dy[0]
    dy_face[-1] = dy[-1]

    # Face-centered velocities (MODFLOW-style).
    vx_face = np.zeros((ny, nx + 1), dtype=float)
    vy_face = np.zeros((ny + 1, nx), dtype=float)

    # Interior x-faces: between columns j-1 and j.
    kx_face = hmean(K[:, :-1], K[:, 1:])
    vx_face[:, 1:nx] = -kx_face * (h[:, 1:] - h[:, :-1]) / dx_face[1:nx]
    # Boundary x-faces: one-sided approximation from nearest cell centers.
    vx_face[:, 0] = -K[:, 0] * (h[:, 1] - h[:, 0]) / dx_face[0]
    vx_face[:, nx] = -K[:, -1] * (h[:, -1] - h[:, -2]) / dx_face[-1]

    # Interior y-faces: between rows i-1 and i.
    ky_face = hmean(K[:-1, :], K[1:, :])
    vy_face[1:ny, :] = -ky_face * (h[1:, :] - h[:-1, :]) / dy_face[1:ny, None]
    # Boundary y-faces: one-sided approximation from nearest cell centers.
    vy_face[0, :] = -K[0, :] * (h[1, :] - h[0, :]) / dy_face[0]
    vy_face[ny, :] = -K[-1, :] * (h[-1, :] - h[-2, :]) / dy_face[-1]

    # Align face outputs with MODFLOW budget convention:
    # - QXX is the flow/velocity at each cell's right face (ny, nx)
    # - QYY is the flow/velocity at each cell's "front" face (ny, nx)
    # Using these slices removes the one-cell offset against MT3D/MODFLOW exports.
    qxx = vx_face[:, 1:]
    qyy = vy_face[1:, :]
    # MODFLOW QXX/QYY do not include an external-neighbor exchange on the
    # outer model edge in these arrays; set those outer faces to zero.
    qxx[:, -1] = 0.0
    qyy[-1, :] = 0.0

    if output_location == "face":
        return qxx, qyy

    # Convert MODFLOW-style per-cell face values to cell-centered values.
    vx_cell = np.empty((ny, nx), dtype=float)
    vy_cell = np.empty((ny, nx), dtype=float)

    vx_cell[:, 0] = qxx[:, 0]
    vx_cell[:, 1:] = 0.5 * (qxx[:, :-1] + qxx[:, 1:])

    vy_cell[0, :] = qyy[0, :]
    vy_cell[1:, :] = 0.5 * (qyy[:-1, :] + qyy[1:, :])
    return vx_cell, vy_cell


def run_modflow(mf2005_config):
    """
    Run MODFLOW simulation using flopy.
    
    Parameters
    ----------
    mf2005_config : dict
        Configuration dictionary containing:
        - lx, ly, lz : Domain dimensions
        - dx, dy, dz : Grid spacing
        - grad_head : Hydraulic head gradient
        - model_ws : Model workspace directory
        - modflow_exe : Path to MODFLOW executable
        - model_name : Model name
        - outftl_name : Output file name
        - logk_field : Log-transformed hydraulic conductivity field
    
    Returns
    -------
    return_msg : str
        Status message indicating success/failure and elapsed time.
    """
    # Import packages inside function to avoid top-level dependency
    import os
    import time
    import flopy
    
    # Set LD_LIBRARY_PATH if needed for MODFLOW executable
    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    # Initialize the model parameters
    lx = mf2005_config['lx']
    ly = mf2005_config['ly']
    lz = mf2005_config['lz']
    dx = mf2005_config['dx']
    dy = mf2005_config['dy']
    dz = mf2005_config['dz']
    x = np.arange(lx)
    delta_h = mf2005_config['grad_head'] * np.abs(x[-1] - x[0])
    # delta_h = mf2005_config['grad_head'] * lx
    model_ws = str(mf2005_config['model_ws'])
    modflow_exe = str(mf2005_config['modflow_exe'])
    model_name = str(mf2005_config['model_name'])
    outftl_name  = str(mf2005_config['outftl_name'])
    k_field = np.exp(mf2005_config['logk_field'])

    # Initialize flopy
    mf = flopy.modflow.Modflow(model_name, model_ws = model_ws, exe_name = modflow_exe)

    # Add DIS package to the MODFLOW model
    ztop = 0.
    zbot = np.zeros((lz, ly, lx))
    for i in range(lz):
        zbot[i, :, :] = ztop - dz*(i+1)

    dis = flopy.modflow.ModflowDis(mf, lz, ly, lx, delr = dy, delc = dx, top = ztop, botm = zbot, perlen = 1)

    # Variables for the BAS package
    ibound = np.ones((lz, ly, lx), dtype = np.int32)
    ibound[:, :,  0] = -1
    ibound[:, :, -1] = -1

    strt = np.zeros((lz, ly, lx), dtype = np.float32)
    for i in range(lz):
        for j in range(ly):
            strt[i, j, :] = np.linspace(delta_h, 0, num = lx)

    bas = flopy.modflow.ModflowBas(mf, ibound = ibound, strt = strt)

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(mf, hk = k_field, vka = 1.)

    # Add OC package to the MODFLOW model
    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf, hclose=1e-10, rclose=1e-10, mxiter=2000, iter1=1000)

    # Add LMT package to the MODFLOW model
    lmt = flopy.modflow.ModflowLmt(mf, output_file_header = 'extended',
                                    output_file_format = 'formatted',
                                    output_file_name = outftl_name)

    # Write the MODFLOW model input files
    mf.write_input()
    
    # Run the MODFLOW model
    start_time = time.time()
    is_successful, buff = mf.run_model(silent=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    status = "successfully" if is_successful else "unsuccessfully"
    return_msg = f'MODFLOW ran {status} after {np.round(elapsed_time, 4)} seconds.'

    return return_msg

def read_mt3d_file(filepath):
    """
    Reads a velocity field file with a specific mixed format.

    Args:
        filepath (str): The path to the input file.

    Returns:
        dict: A dictionary containing the parsed variables.
    """
    # Import re module inside function to avoid top-level dependency
    import re
    
    variables = {}
    with open(filepath, 'r') as f:
        # Read the main header line and the next line of integers
        main_header = f.readline().strip()
        _ = f.readline().strip().split() # Read and discard the integer line

        variables['header'] = main_header

        # --- This pattern of reading 2D arrays is repeated ---
        def read_2d_array(file_handle):
            data_list = []
            while True:
                # Store the current position in case we read too far
                last_pos = file_handle.tell()
                line = file_handle.readline()

                
                # Check for end of file
                if not line:
                    break
                
                # The dimension line is the delimiter between variable blocks
                if re.match(r'^\s*(\d+\s+){4}\d+\s*$', line):
                    dims = [int(x) for x in line.split()]
                    break
                
                # If it's not a dimension line, rewind and read the header of the next block
                if "'" in line:
                    file_handle.seek(last_pos)
                    dims = None # Signal that we didn't find a dimension line
                    break
                    
                data_list.extend(line.strip().split())
            
            # If a dimension line was found, reshape the array
            if dims:
                rows, cols = dims[2], dims[3]
                return np.array(data_list, dtype=float).reshape((rows, cols))
            return None # Should not happen in a well-formed file for these vars

        # Read the first variable block: THKSAT
        var_header = f.readline().strip()
        variable_name = var_header.strip("' ")
        variables[variable_name] = read_2d_array(f)
        
        # Read the second variable block: QXX
        var_header = f.readline().strip()
        variable_name = var_header.strip("' ")
        variables[variable_name] = read_2d_array(f)
        
        # Read the third variable block: QYY
        var_header = f.readline().strip()
        variable_name = var_header.strip("' ")
        variables[variable_name] = read_2d_array(f)

        # Read the fourth variable block: CNH
        var_header_line = f.readline()
        match = re.search(r"'(.*?)'\s+(\d+)", var_header_line)
        if match:
            variable_name = match.group(1).strip()
            num_records = int(match.group(2))
        else:
            raise ValueError(f"Failed to parse CNH header line: {var_header_line}")

        _ = f.readline()

        # --- FIX IS HERE ---
        cnh_data = []
        for i in range(num_records):
            line = f.readline()

            # Safely handle the end of file or blank lines
            if not line or not line.strip():
                print(f"Warning: Expected {num_records} records for '{variable_name}', but found a blank line or EOF after {i} records.")
                break # Stop reading this block

            line_parts = line.strip().split()
            
            # Defensive check for expected number of columns
            if len(line_parts) < 4:
                print(f"Warning: Skipping malformed line in '{variable_name}' data: {line.strip()}")
                continue # Skip this line and go to the next

            cnh_data.append([
                int(line_parts[0]),
                int(line_parts[1]),
                int(line_parts[2]),
                float(line_parts[3])
            ])
        variables[variable_name] = np.array(cnh_data)
        # --- END OF FIX ---

    return variables