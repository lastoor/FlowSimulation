#!/usr/bin/env python
"""
Unit tests for grf_generation.py
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import grf_generation


class TestFbmGenerator(unittest.TestCase):
    """Test fBm generator functions."""

    def test_compute_c(self):
        """Test scale factor computation."""
        # Simple case: l=3, H=0.5, std_target=1.0
        l = 3
        H = 0.5
        std_target = 1.0
        c = grf_generation.compute_c(l, H, std_target)
        # Compute expected manually
        coords = np.arange(l) - (l - 1) / 2
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        r = np.sqrt(X**2 + Y**2)
        avg_r2H = np.mean(r**(2 * H))
        expected = std_target**2 / (2 * avg_r2H)
        self.assertAlmostEqual(c, expected)

        # Larger grid, different H
        l = 10
        H = 0.25
        std_target = 2.0
        c = grf_generation.compute_c(l, H, std_target)
        coords = np.arange(l) - (l - 1) / 2
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        r = np.sqrt(X**2 + Y**2)
        avg_r2H = np.mean(r**(2 * H))
        expected = std_target**2 / (2 * avg_r2H)
        self.assertAlmostEqual(c, expected)

    def test_PrefbmCov(self):
        """Test PrefbmCov covariance function."""
        # Test alpha in (0, 1.5]
        cov = grf_generation.PrefbmCov(alpha=0.5)
        self.assertEqual(cov.alpha, 0.5)
        self.assertTrue(cov.is_even)
        # At r=0
        self.assertAlmostEqual(cov([0, 0]), 1 - 0.5/2)  # c0
        # At r=0.5
        r = 0.5
        expected = cov.c0 - r**0.5 + cov.c2 * r**2
        self.assertAlmostEqual(cov([r, 0]), expected)
        # At r=1.0
        self.assertAlmostEqual(cov([1, 0]), 0.0)
        # At r=2.0 (beyond cutoff)
        self.assertAlmostEqual(cov([2, 0]), 0.0)

        # Test alpha in (1.5, 2)
        cov2 = grf_generation.PrefbmCov(alpha=1.8)
        self.assertEqual(cov2.alpha, 1.8)
        # At r=0
        self.assertAlmostEqual(cov2([0, 0]), 1 - 1.8/6 - 1.8**2/6)
        # At r=1.0
        expected = cov2.c0 - 1.0**1.8 + cov2.c2 * 1.0**2
        self.assertAlmostEqual(cov2([1, 0]), expected)
        # At r=1.5 (between 1 and 2)
        r = 1.5
        expected = 1.8*(2-1.8)/18 * (2-r)**3 / r
        self.assertAlmostEqual(cov2([r, 0]), expected)
        # At r=2.0
        self.assertAlmostEqual(cov2([2, 0]), 0.0)

        # Invalid alpha raises ValueError when calling covariance
        cov_invalid = grf_generation.PrefbmCov(alpha=2.5)
        with self.assertRaises(ValueError):
            cov_invalid([1.0, 0.0])
        cov_invalid2 = grf_generation.PrefbmCov(alpha=0)
        with self.assertRaises(ValueError):
            cov_invalid2([1.0, 0.0])

    def test_GaussianRandomFieldCircEmbed_init(self):
        """Test Gaussian random field initialization."""
        mean = np.zeros((5, 5))
        # Simple covariance: exponential
        class SimpleCov:
            is_even = False
            def __call__(self, x):
                return np.exp(-np.linalg.norm(x))
        cov = SimpleCov()
        pts = [np.arange(5), np.arange(5)]
        grf = grf_generation.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=0)
        self.assertIsNotNone(grf.data)
        Λ, shape_emb = grf.data
        self.assertEqual(len(shape_emb), 2)
        # shape_emb should be at least 5 in each dimension
        self.assertGreaterEqual(shape_emb[0], 5)
        self.assertGreaterEqual(shape_emb[1], 5)

    def test_GaussianRandomFieldCircEmbed_sample(self):
        """Test Gaussian random field sampling."""
        np.random.seed(42)
        mean = np.zeros((3, 3))
        class SimpleCov:
            is_even = False
            def __call__(self, x):
                return np.exp(-np.linalg.norm(x))
        cov = SimpleCov()
        pts = [np.arange(3), np.arange(3)]
        grf = grf_generation.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=0)
        sample = grf.sample()
        self.assertEqual(sample.shape, (3, 3))
        # Mean should be close to zero (but not exactly due to random)
        self.assertAlmostEqual(np.mean(sample), 0.0, delta=0.5)
        # Variance should be finite
        self.assertGreater(np.var(sample), 0)

    def test_fBmGenerator_init(self):
        """Test fBm generator initialization."""
        fbm = grf_generation.fBmGenerator(c=2.0, alpha=0.5, num_pt=32, dim=2, minpadding=0)
        self.assertEqual(fbm.c, 2.0)
        self.assertEqual(fbm.alpha, 0.5)
        self.assertEqual(fbm.num_pt, 32)
        self.assertEqual(len(fbm.pts), 2)  # 2D
        self.assertEqual(fbm.pts[0][0], 0.0)
        self.assertEqual(fbm.pts[0][-1], 31/32)  # normalized to [0,1)
        self.assertIsNotNone(fbm._prefbm_gen)
        self.assertIsNotNone(fbm.cov_c0)
        self.assertIsNotNone(fbm.cov_c2)
        self.assertIsNotNone(fbm._σ_trend)

    def test_fBmGenerator_sample(self):
        """Test fBm field sampling."""
        np.random.seed(123)
        fbm = grf_generation.fBmGenerator(c=1.0, alpha=0.7, num_pt=16, dim=2, minpadding=0)
        field = fbm.sample()
        # Shape should be (16, 16)
        self.assertEqual(field.shape, (16, 16))
        # Field should be real-valued
        self.assertTrue(np.isrealobj(field))
        # Variance should be positive
        self.assertGreater(np.var(field), 0)

    def test_generate_logk_field(self):
        """Test logk field generation wrapper."""
        np.random.seed(99)
        fbm = grf_generation.fBmGenerator(c=1.0, alpha=0.5, num_pt=8, dim=2, minpadding=0)
        logk_field, msg = grf_generation.generate_logk_field(fbm)
        self.assertIsNotNone(logk_field)
        self.assertEqual(logk_field.shape, (8, 8))
        self.assertIn('successfully', msg)
        # Test error case (mock exception)
        class BrokenFBM:
            def sample(self):
                raise RuntimeError('Test error')
        broken = BrokenFBM()
        logk_field, msg = grf_generation.generate_logk_field(broken)
        self.assertIsNone(logk_field)
        self.assertIn('unsuccessfully', msg)


if __name__ == '__main__':
    unittest.main()