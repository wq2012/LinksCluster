"""Unit tests for Subcluster class."""

import unittest
import numpy as np
from linkscluster import subcluster
from linkscluster import utils


class TestSubcluster(unittest.TestCase):
  """Tests for Subcluster class."""

  def test_initialization(self):
    v = utils.l2_normalize(np.array([1.0, 2.0, 3.0]))
    sc = subcluster.Subcluster(
        subcluster_id=0,
        cluster_id=1,
        vector=v,
        vector_idx=0,
        tc=0.6,
        index=0,
        use_theoretical_norm=False)
    self.assertEqual(sc.subcluster_id, 0)
    self.assertEqual(sc.id, 0)
    self.assertEqual(sc.cluster_id, 1)
    self.assertEqual(sc.count, 1)
    self.assertEqual(sc.k, 1)
    self.assertEqual(sc.vector_indices, [0])
    self.assertAlmostEqual(np.linalg.norm(sc.centroid), 1.0, places=6)
    np.testing.assert_allclose(sc.centroid, v)
    np.testing.assert_allclose(sc.mu_hat, v)

  def test_add_vector(self):
    v1 = utils.l2_normalize(np.array([1.0, 0.0]))
    v2 = utils.l2_normalize(np.array([0.0, 1.0]))
    sc = subcluster.Subcluster(
        subcluster_id=0,
        cluster_id=0,
        vector=v1,
        vector_idx=0,
        tc=0.6,
        index=0,
        use_theoretical_norm=False)
    sc.add_vector(v2, 1)
    self.assertEqual(sc.count, 2)
    self.assertEqual(sc.vector_indices, [0, 1])
    self.assertAlmostEqual(np.linalg.norm(sc.centroid), 1.0, places=6)
    expected = utils.l2_normalize(v1 + v2)
    np.testing.assert_allclose(sc.centroid, expected)

  def test_merge_with(self):
    v1 = utils.l2_normalize(np.array([1.0, 0.0]))
    v2 = utils.l2_normalize(np.array([0.0, 1.0]))
    sc1 = subcluster.Subcluster(0, 0, v1, 0, tc=0.6, index=0)
    sc2 = subcluster.Subcluster(1, 0, v2, 1, tc=0.6, index=1)
    sc1.merge_with(sc2)
    self.assertEqual(sc1.count, 2)
    self.assertEqual(sc1.vector_indices, [0, 1])
    expected = utils.l2_normalize(v1 + v2)
    np.testing.assert_allclose(sc1.centroid, expected)

  def test_theoretical_norm(self):
    v = utils.l2_normalize(np.array([1.0, 0.0, 0.0]))
    sc = subcluster.Subcluster(
        0, 0, v, 0, tc=0.6, index=0, use_theoretical_norm=True)
    self.assertAlmostEqual(np.linalg.norm(sc.centroid), 1.0, places=6)


if __name__ == "__main__":
  unittest.main()
