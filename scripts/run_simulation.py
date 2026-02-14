#!/usr/bin/env python
"""
Entry point script for running the simulation.
"""

import os
import sys

# Add libgfortran.so.5 and libstdc++.so.6 in /apps/generic/gcc/13.3.0/lib64 as an environment variable
os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Add the parent directory to sys.path to import simulation package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.main import main

if __name__ == '__main__':
    main()