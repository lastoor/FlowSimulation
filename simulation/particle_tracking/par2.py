"""
PAR2: run external PAR2 executable for particle tracking.
"""

import subprocess
import time
from pathlib import Path

import numpy as np


def run_par2(path_par2, path_p2input, path_p2output):
    """
    Run the PAR2 executable; clean output dirs (debut_trajectory, all_trajectory)
    before running. path_p2output is the base output directory (Path-like).
    Returns (returncode, message).
    """
    path_debuttraj = path_p2output / "debut_trajectory"
    path_alltraj = path_p2output / "all_trajectory"

    if path_debuttraj.exists() and any(path_debuttraj.iterdir()):
        for file in path_debuttraj.iterdir():
            if file.is_file():
                file.unlink()

    if path_alltraj.exists() and any(path_alltraj.iterdir()):
        for file in path_alltraj.iterdir():
            if file.is_file():
                file.unlink()

    start_time = time.time()
    proc = subprocess.run(
        [path_par2, path_p2input / "par2_config.yaml"],
        capture_output=False,
        text=True,
    )
    elapsed_time = time.time() - start_time

    if proc.returncode == 0:
        return_msg = f"PAR2 ran successfully after {np.round(elapsed_time, 4)} seconds."
    elif proc.returncode == 10:
        return_msg = f"PAR2 ran successfully after {np.round(elapsed_time, 4)} seconds, although no debut."
    elif proc.returncode == 11:
        return_msg = f"PAR2 ran successfully after {np.round(elapsed_time, 4)} seconds, although half debut."
    else:
        return_msg = f"PAR2 ran unsuccessfully after {np.round(elapsed_time, 4)} seconds. Return code: {proc.returncode}."

    return proc.returncode, return_msg
