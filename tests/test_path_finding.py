#!/usr/bin/env python
"""
Unit tests for path_finding.py
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import path_finding


class TestPathFinding(unittest.TestCase):
    """Test path finding algorithms."""

    def test_generate_rectangular_array(self):
        """Test rectangular grid generation."""
        # 2D case
        vertex_pair = [(0, 0), (2, 3)]
        grid = path_finding.generate_rectangular_array(vertex_pair)
        expected_shape = (3, 4, 2)  # (x_range, y_range, dim)
        self.assertEqual(grid.shape, expected_shape)

        # Check coordinates
        self.assertEqual(grid[0, 0, 0], 0)
        self.assertEqual(grid[0, 0, 1], 0)
        self.assertEqual(grid[2, 3, 0], 2)
        self.assertEqual(grid[2, 3, 1], 3)

        # 3D case (though not used in project, function supports any dim)
        vertex_pair_3d = [(0, 0, 0), (1, 1, 1)]
        grid_3d = path_finding.generate_rectangular_array(vertex_pair_3d)
        self.assertEqual(grid_3d.shape, (2, 2, 2, 3))

    def test_is_path_valid_valid(self):
        """Test path validation for valid path."""
        # Straight line path within box
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        box = (0, 3, 0, 0)  # x0, x1, y0, y1
        len_min = 2.0
        self.assertTrue(path_finding.is_path_valid(path, box, len_min))

        # Path longer than minimum
        path2 = np.array([[0, 0], [0, 1], [0, 2]])
        box2 = (0, 0, 0, 2)
        self.assertTrue(path_finding.is_path_valid(path2, box2, 1.0))

    def test_is_path_valid_invalid_box(self):
        """Test path validation for path outside box."""
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        box = (0, 2, 0, 0)  # x max is 2, path goes to 3
        len_min = 1.0
        self.assertFalse(path_finding.is_path_valid(path, box, len_min))

        # y out of bounds
        path2 = np.array([[0, 0], [0, 1], [0, 2]])
        box2 = (0, 0, 0, 1)
        self.assertFalse(path_finding.is_path_valid(path2, box2, 1.0))

    def test_is_path_valid_too_short(self):
        """Test path validation for path shorter than minimum."""
        path = np.array([[0, 0], [1, 0]])
        box = (0, 1, 0, 0)
        len_min = 2.0  # path length is 1.0
        self.assertFalse(path_finding.is_path_valid(path, box, len_min))

    def test_dijkstra_2d_least_resistance_simple(self):
        """Test Dijkstra on a simple uniform conductivity field."""
        # Uniform conductivity
        K = np.ones((5, 5), dtype=float)
        start_region = [(2, 0)]  # middle left
        end_region = [(2, 4)]    # middle right

        path, total_resistance, start_node, end_node, resistance_map = \
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region, average='harmonic')

        # Should be a straight horizontal line
        self.assertEqual(len(path), 5)  # from col 0 to col 4
        # Check that path moves rightward
        for i in range(1, len(path)):
            self.assertEqual(path[i, 1], path[i-1, 1])  # same row
            self.assertEqual(path[i, 0], path[i-1, 0] + 1)  # col increments by 1

        # Total resistance should be sum of edge resistances
        # Each edge: length=1, k_bond=1 (harmonic mean of 1 and 1 = 1), resistance = 1/1 = 1
        # 4 edges total
        self.assertAlmostEqual(float(total_resistance), 4.0)

        # Start and end nodes should match
        self.assertEqual(start_node, (2, 0))
        self.assertEqual(end_node, (2, 4))

        # Resistance map should have finite values along path
        self.assertLess(resistance_map[2, 0], np.inf)
        self.assertLess(resistance_map[2, 4], np.inf)

    def test_dijkstra_2d_least_resistance_arithmetic(self):
        """Test Dijkstra with arithmetic averaging."""
        K = np.ones((3, 3), dtype=float)
        start_region = [(1, 0)]
        end_region = [(1, 2)]

        path, total_resistance, start_node, end_node, resistance_map = \
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region, average='arithmetic')

        # Still straight line
        self.assertEqual(len(path), 3)
        self.assertEqual(start_node, (1, 0))
        self.assertEqual(end_node, (1, 2))
        # Arithmetic mean also 1, resistance per edge = 1
        self.assertAlmostEqual(float(total_resistance), 2.0)

    def test_dijkstra_2d_least_resistance_high_low_conductivity(self):
        """Test Dijkstra prefers high conductivity regions."""
        # Create a field with a high conductivity channel
        K = np.ones((5, 5), dtype=float) * 0.1  # low conductivity background
        K[:, 2] = 10.0  # vertical high conductivity column

        start_region = [(0, 0)]
        end_region = [(4, 4)]

        path, total_resistance, _, _, _ = \
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region)

        # Path should go through high conductivity column
        # Expect path: (0,0) -> ... -> column 2, then down, then to (4,4)
        # We'll just check that path contains some points in column 2
        col_values = path[:, 0]
        self.assertTrue(2 in col_values)

    def test_dijkstra_2d_least_resistance_zero_conductivity(self):
        """Test Dijkstra with zero conductivity (infinite resistance)."""
        K = np.ones((3, 3), dtype=float)
        K[1, 1] = 0.0  # Blocked cell

        start_region = [(0, 1)]
        end_region = [(2, 1)]

        path, total_resistance, start_node, end_node, resistance_map = \
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region)

        # Should find a path around the blocked cell
        self.assertIsNotNone(path)
        self.assertLess(total_resistance, np.inf)
        # Path should avoid (1,1)
        self.assertNotIn((1, 1), [tuple(p) for p in path])

    def test_dijkstra_2d_least_resistance_no_path(self):
        """Test Dijkstra when no path exists (isolated by zero conductivity)."""
        K = np.ones((3, 3), dtype=float)
        K[:, 1] = 0.0  # Vertical barrier

        start_region = [(0, 0)]
        end_region = [(0, 2)]  # Need to cross column 1

        path, total_resistance, start_node, end_node, resistance_map = \
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region)

        # Should return empty path? Actually function returns path as empty? Let's see.
        # Based on code, if actual_end_node is None, path is empty? Let's check.
        # The function returns path array with shape (0, 2)? We'll just check total_resistance is inf
        self.assertEqual(len(path), 0)
        self.assertEqual(total_resistance, np.inf)

    def test_dijkstra_invalid_average(self):
        """Test Dijkstra with invalid average type raises ValueError."""
        K = np.ones((3, 3), dtype=float)
        start_region = [(0, 0)]
        end_region = [(2, 2)]

        with self.assertRaises(ValueError):
            path_finding.dijkstra_2d_least_resistance(K, start_region, end_region, average='invalid')


if __name__ == '__main__':
    unittest.main()