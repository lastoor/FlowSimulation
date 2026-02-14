#!/usr/bin/env python
"""
Unit tests for config.py
"""

import sys
import os
import unittest
from unittest.mock import patch
from argparse import Namespace

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import config


class TestConfig(unittest.TestCase):
    """Test configuration parsing and creation."""

    def test_parse_args(self):
        """Test argument parsing with required arguments."""
        test_args = [
            '--path_workdir', '/tmp/test',
            '--num_mc', '10', '20',
            '--field_size', '100', '200',
            '--hurst', '0.5',
            '--minpadding', '128',
            '--c', '2.0',
            '--std', '5.0'
        ]
        with patch.object(sys, 'argv', ['test_script.py'] + test_args):
            args = config.parse_args()
            self.assertEqual(args.path_workdir, '/tmp/test')
            self.assertEqual(args.num_mc, [10, 20])
            self.assertEqual(args.field_size, [100, 200])
            self.assertEqual(args.hurst, 0.5)
            self.assertEqual(args.minpadding, 128)
            self.assertEqual(args.c, 2.0)
            self.assertEqual(args.std, 5.0)

    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        test_args = [
            '--path_workdir', '/tmp/test',
            '--num_mc', '5',
            '--field_size', '50',
            '--hurst', '0.25'
        ]
        with patch.object(sys, 'argv', ['test_script.py'] + test_args):
            args = config.parse_args()
            self.assertEqual(args.path_workdir, '/tmp/test')
            self.assertEqual(args.num_mc, [5])
            self.assertEqual(args.field_size, [50])
            self.assertEqual(args.hurst, 0.25)
            self.assertEqual(args.minpadding, 0)  # default is 0
            self.assertIsNone(args.c)
            self.assertIsNone(args.std)

    def test_create_base_config(self):
        """Test base configuration creation."""
        args = Namespace(
            path_workdir='/tmp/test',
            num_mc=[10, 20],
            field_size=[100, 200],
            hurst=0.5,
            minpadding=128,
            c=2.0,
            std=5.0
        )
        config_base = config.create_base_config(args)

        # Check basic structure
        self.assertIn('num_mc', config_base)
        self.assertIn('field_size', config_base)
        self.assertIn('fbm_arg', config_base)
        self.assertIn('headSolver_arg', config_base)
        self.assertIn('particleTracking_arg', config_base)

        # Check values
        self.assertEqual(config_base['num_mc'], [10, 20])
        self.assertEqual(config_base['field_size'], [100, 200])
        self.assertEqual(config_base['fbm_arg']['hurst'], 0.5)
        self.assertEqual(config_base['fbm_arg']['std'], 5.0)
        self.assertEqual(config_base['fbm_arg']['c'], 2.0)
        self.assertEqual(config_base['fbm_arg']['minpadding'], 128)

        # Check derived paths
        self.assertTrue(str(config_base['path_simulation']).endswith('/tmp/test'))
        self.assertIn('log', str(config_base['path_log']))

        # Check particle tracking args
        self.assertIn('density_particle', config_base['particleTracking_arg'])
        self.assertEqual(config_base['particleTracking_arg']['y_policy'], 'stop')

    def test_create_base_config_defaults(self):
        """Test base configuration with default values."""
        args = Namespace(
            path_workdir='/tmp/test',
            num_mc=[5],
            field_size=[50],
            hurst=0.25,
            minpadding=0,
            c=None,
            std=None
        )
        config_base = config.create_base_config(args)

        self.assertEqual(config_base['fbm_arg']['std'], None)
        self.assertEqual(config_base['fbm_arg']['c'], None)
        self.assertEqual(config_base['fbm_arg']['minpadding'], 0)


if __name__ == '__main__':
    unittest.main()