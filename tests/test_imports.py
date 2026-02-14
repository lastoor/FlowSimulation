#!/usr/bin/env python
"""
Test that all modules can be imported.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    import simulation
    import simulation.utils
    import simulation.grf_generation
    import simulation.hydraulic
    import simulation.particle_tracking
    import simulation.path_finding
    import simulation.config
    import simulation.main
    print("All imports successful.")

if __name__ == '__main__':
    test_imports()