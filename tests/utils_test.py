"""Unit tests for utility functions."""

import unittest
import numpy as np
from linkscluster import utils


class TestUtils(unittest.TestCase):
  """Tests for utils module."""

  def test_single_threshold(self):
    tc = 0.6
    # s(1) == tc^2 == 0.36
    s1 = utils.single_threshold(1, tc)
    self.assertAlmostEqual(s1, tc ** 2, places=6)

    # Monotonically increasing with k
    s10 = utils.single_threshold(10, tc)
    s100 = utils.single_threshold(100, tc)
    self.assertGreater(s10, s1)
    self.assertGreater(s100, s10)
    self.assertLess(s100, tc)

    with self.assertRaises(ValueError):
      utils.single_threshold(0, tc)
    with self.assertRaises(ValueError):
      utils.single_threshold(1, 0.0)
    with self.assertRaises(ValueError):
      utils.single_threshold(1, 1.0)

  def test_multi_threshold(self):
    tc = 0.6
    # s(1, 1) == tc^2
    s11 = utils.multi_threshold(1, 1, tc)
    self.assertAlmostEqual(s11, tc ** 2, places=6)

    # s(k, 1) == s(k)
    s10_1 = utils.multi_threshold(10, 1, tc)
    self.assertAlmostEqual(s10_1, utils.single_threshold(10, tc), places=6)

    # Symmetry
    s10_20 = utils.multi_threshold(10, 20, tc)
    s20_10 = utils.multi_threshold(20, 10, tc)
    self.assertAlmostEqual(s10_20, s20_10, places=6)

    # Limit as k1, k2 -> infty is 1
    s_large = utils.multi_threshold(100000, 100000, tc)
    self.assertAlmostEqual(s_large, 1.0, places=3)

    with self.assertRaises(ValueError):
      utils.multi_threshold(0, 1, tc)
    with self.assertRaises(ValueError):
      utils.multi_threshold(1, -1, tc)

  def test_anisotropic_threshold(self):
    tc = 0.6
    tp = 0.9
    # s_tilde(1, 1) == tc^2
    s11 = utils.anisotropic_threshold(1, 1, tc, tp)
    self.assertAlmostEqual(s11, tc ** 2, places=6)

    # limit as k1, k2 -> infty is tp
    s_large = utils.anisotropic_threshold(100000, 100000, tc, tp)
    self.assertAlmostEqual(s_large, tp, places=3)

    # tp=1.0 or None falls back to multi_threshold
    s_iso1 = utils.anisotropic_threshold(5, 5, tc, 1.0)
    s_iso2 = utils.anisotropic_threshold(5, 5, tc, None)
    s_multi = utils.multi_threshold(5, 5, tc)
    self.assertAlmostEqual(s_iso1, s_multi, places=6)
    self.assertAlmostEqual(s_iso2, s_multi, places=6)

    # Invalid tp < tc^2
    with self.assertRaises(ValueError):
      utils.anisotropic_threshold(1, 1, tc, 0.2)

  def test_l2_normalize(self):
    v = np.array([3.0, 4.0])
    v_norm = utils.l2_normalize(v)
    self.assertAlmostEqual(np.linalg.norm(v_norm), 1.0, places=6)

    mat = np.array([[3.0, 4.0], [1.0, 0.0]])
    mat_norm = utils.l2_normalize(mat)
    self.assertAlmostEqual(np.linalg.norm(mat_norm[0]), 1.0, places=6)
    self.assertAlmostEqual(np.linalg.norm(mat_norm[1]), 1.0, places=6)

    zero_v = np.array([0.0, 0.0])
    res = utils.l2_normalize(zero_v)
    self.assertEqual(res.shape, (2,))

  def test_enforce_ordered_labels(self):
    labels = np.array([5, 5, 2, 8, 2])
    ordered = utils.enforce_ordered_labels(labels)
    expected = np.array([0, 0, 1, 2, 1])
    np.testing.assert_array_equal(ordered, expected)

  def test_compute_accuracy(self):
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    pred_labels = np.array([1, 1, 0, 0, 2, 2])  # permutation
    acc = utils.compute_accuracy(true_labels, pred_labels)
    self.assertAlmostEqual(acc, 1.0, places=6)

    pred_wrong = np.array([0, 0, 1, 2, 2, 2])
    acc_wrong = utils.compute_accuracy(true_labels, pred_wrong)
    self.assertAlmostEqual(acc_wrong, 5.0 / 6.0, places=6)

    # Empty
    self.assertAlmostEqual(
        utils.compute_accuracy(np.array([]), np.array([])), 1.0)

    # Mismatch shape
    with self.assertRaises(ValueError):
      utils.compute_accuracy(np.array([1, 2]), np.array([1]))


if __name__ == "__main__":
  unittest.main()
