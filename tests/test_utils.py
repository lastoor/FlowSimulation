#!/usr/bin/env python
"""
Unit tests for utils.py
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import utils


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = utils.get_logger(name='test_logger', path_logfile='/tmp/test.log')
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_logger')
        self.assertEqual(logger.level, logging.INFO)
        # Should have handlers
        self.assertTrue(logger.hasHandlers())

    @patch('subprocess.run')
    def test_get_gpu_usage_success(self, mock_run):
        """Test GPU usage query success."""
        mock_run.return_value.stdout = '100, 8000\n'
        mock_run.return_value.stderr = ''
        result = utils.get_gpu_usage()
        # Format is "Memory: 100MB used, 8,000MB free." (with comma separator)
        self.assertIn('Memory: 100MB used', result)
        self.assertIn('8,000MB free.', result)
        mock_run.assert_called_once_with(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
            stdout=ANY, stderr=ANY, text=True
        )

    @patch('subprocess.run')
    def test_get_gpu_usage_error(self, mock_run):
        """Test GPU usage query error."""
        mock_run.side_effect = Exception('Command not found')
        result = utils.get_gpu_usage()
        self.assertIn('GPU info error:', result)

    @patch('subprocess.run')
    def test_log_gpu_usage(self, mock_run):
        """Test GPU logging function."""
        mock_run.return_value.stdout = '2025-01-01 12:00:00, 50, 100, 8000\n'
        mock_run.return_value.stderr = ''

        # Mock event to stop after one iteration
        mock_event = MagicMock()
        mock_event.is_set.side_effect = [False, True]  # First call returns False, second True

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            utils.log_gpu_usage(path_logfile='/tmp/gpu.log', interval=0.001, event_stop=mock_event)

        # Check that subprocess.run was called
        self.assertEqual(mock_run.call_count, 1)
        # Check file write
        mock_file.assert_called_with('/tmp/gpu.log', 'w')
        handle = mock_file()
        # First line should be header
        handle.write.assert_any_call("timestamp, gpu_utilization [%], memory_used [MB], memory_free [MB]\n")
        # Then data line
        handle.write.assert_any_call('2025-01-01 12:00:00, 50, 100, 8000\n')

    def test_euclidean_length_cpu(self):
        """Test Euclidean length calculation on CPU (no GPU)."""
        # Simple 2D trajectory: straight line from (0,0) to (3,4)
        trajectory = np.array([[0, 0], [3, 4]], dtype=np.float64)
        length = utils.euclidean_length(trajectory)
        expected = 5.0  # sqrt(3^2 + 4^2)
        self.assertAlmostEqual(length, expected, places=6)

        # Multi-segment trajectory
        trajectory = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        length = utils.euclidean_length(trajectory)
        expected = 2.0  # 1 + 1
        self.assertAlmostEqual(length, expected, places=6)

        # Single point (should return 0)
        trajectory = np.array([[1, 2]], dtype=np.float64)
        length = utils.euclidean_length(trajectory)
        self.assertEqual(length, 0.0)

    @patch('simulation.utils._cupy_available', False)
    def test_euclidean_length_no_cupy(self):
        """Test Euclidean length when CuPy is not available."""
        trajectory = np.array([[0, 0], [3, 4]], dtype=np.float64)
        length = utils.euclidean_length(trajectory)
        self.assertAlmostEqual(length, 5.0, places=6)

    # Note: Testing with CuPy requires GPU environment; we skip for unit tests

    def test_euclidean_length_3d(self):
        """Test Euclidean length with 3D trajectory."""
        trajectory = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]], dtype=np.float64)
        length = utils.euclidean_length(trajectory)
        expected = 1 + 2 + 3  # segment lengths
        self.assertAlmostEqual(length, expected, places=6)


if __name__ == '__main__':
    unittest.main()