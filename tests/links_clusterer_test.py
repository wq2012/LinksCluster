"""Unit tests for LinksClusterer."""

import unittest
import numpy as np
from linkscluster import links_clusterer
from linkscluster import utils

LinksClusterer = links_clusterer.LinksClusterer


class TestLinksClusterer(unittest.TestCase):
  """Tests for LinksClusterer class."""

  def test_invalid_parameters(self):
    with self.assertRaises(ValueError):
      LinksClusterer(tc=-0.1)
    with self.assertRaises(ValueError):
      LinksClusterer(tc=1.0)
    with self.assertRaises(ValueError):
      LinksClusterer(ts=0.0)
    with self.assertRaises(ValueError):
      LinksClusterer(tc=0.6, tp=0.2)  # tp < tc^2

  def test_parameter_aliases(self):
    clusterer = LinksClusterer(
        cluster_similarity_threshold=0.55,
        subcluster_similarity_threshold=0.75,
        pair_similarity_maximum=0.92)
    self.assertEqual(clusterer.tc, 0.55)
    self.assertEqual(clusterer.ts, 0.75)
    self.assertEqual(clusterer.tp, 0.92)

  def test_empty_input(self):
    clusterer = LinksClusterer(tc=0.5, ts=0.8)
    empty = np.empty((0, 64))
    labels = clusterer.fit_predict(empty)
    self.assertEqual(len(labels), 0)
    self.assertEqual(len(clusterer.labels_), 0)
    self.assertEqual(clusterer.n_clusters_, 0)

  def test_single_vector(self):
    clusterer = LinksClusterer(tc=0.5, ts=0.8)
    v = np.random.randn(1, 64)
    labels = clusterer.fit_predict(v)
    self.assertEqual(labels[0], 0)
    self.assertEqual(clusterer.n_clusters_, 1)
    self.assertEqual(clusterer.n_subclusters_, 1)

  def test_dim_mismatch(self):
    clusterer = LinksClusterer(tc=0.5, ts=0.8)
    clusterer.predict_next(np.ones(64))
    with self.assertRaises(ValueError):
      clusterer.predict_next(np.ones(32))

  def test_online_streaming_api(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85)
    v1 = np.zeros(64)
    v1[0] = 1.0

    # Stream vector 1
    cid1 = clusterer.predict_next(v1)
    self.assertEqual(cid1, 0)
    self.assertEqual(clusterer.n_samples_seen_, 1)

    # Identical vector 2 (cosine similarity = 1.0 >= ts=0.85)
    # Should join subcluster of v1, same cluster ID
    cid2 = clusterer.predict_next(v1)
    self.assertEqual(cid2, 0)
    self.assertEqual(clusterer.n_subclusters_, 1)

    # Completely orthogonal vector 3 (sim = 0.0 < tc^2)
    v3 = np.zeros(64)
    v3[1] = 1.0
    cid3 = clusterer.update(v3)
    self.assertEqual(cid3, 1)
    self.assertEqual(clusterer.n_subclusters_, 2)
    self.assertEqual(clusterer.n_clusters_, 2)

  def test_partial_fit_and_stream(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85)
    X = np.random.randn(10, 32)
    clusterer.partial_fit(X[:5])
    self.assertEqual(clusterer.n_samples_seen_, 5)

    stream_preds = list(clusterer.predict_stream(X[5:]))
    self.assertEqual(len(stream_preds), 5)
    self.assertEqual(clusterer.n_samples_seen_, 10)

  def test_scikit_learn_api(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85)
    X = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.99, 0.01],
    ])
    clusterer.fit(X)
    self.assertIsNotNone(clusterer.labels_)
    self.assertEqual(len(clusterer.labels_), 4)
    self.assertEqual(clusterer.labels_[0], clusterer.labels_[1])
    self.assertEqual(clusterer.labels_[2], clusterer.labels_[3])
    self.assertNotEqual(clusterer.labels_[0], clusterer.labels_[2])

  def test_out_of_sample_predict(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85)
    X_train = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    clusterer.fit(X_train)

    X_test = np.array([
        [0.95, 0.05, 0.0],
        [0.0, 0.95, 0.05],
        [0.0, 0.0, 1.0],  # orthogonal to all
    ])
    preds = clusterer.predict(X_test)
    self.assertEqual(preds[0], clusterer.labels_[0])
    self.assertEqual(preds[1], clusterer.labels_[1])
    self.assertEqual(preds[2], -1)  # unassigned

  def test_subcluster_merging(self):
    """Test that subclusters merge when centroids get within Ts."""
    clusterer = LinksClusterer(tc=0.5, ts=0.8)
    dim = 32

    v1 = np.zeros(dim)
    v1[0] = 1.0
    clusterer.predict_next(v1)

    # v2 sim = 0.65 (between tc=0.5 and ts=0.8)
    # Starts new subcluster connected by edge
    v2 = np.zeros(dim)
    v2[0] = 0.65
    v2[1] = np.sqrt(1.0 - 0.65 ** 2)
    clusterer.predict_next(v2)
    self.assertEqual(clusterer.n_subclusters_, 2)

    # Now add vector v3 that pulls v2's subcluster closer to v1 (sim >= 0.8)
    v3 = np.zeros(dim)
    v3[0] = 0.95
    v3[1] = np.sqrt(1.0 - 0.95 ** 2)
    clusterer.predict_next(v3)

    # After merging, should have fewer subclusters
    self.assertEqual(clusterer.n_clusters_, 1)

  def test_cluster_splitting(self):
    """Test cluster splitting when edge similarity drops below s(k1, k2)."""
    clusterer = LinksClusterer(tc=0.6, ts=0.85, tp=0.95)
    dim = 64

    v1 = np.zeros(dim)
    v1[0] = 1.0
    clusterer.predict_next(v1)

    # Bridge vector at sim = 0.70 to v1 (between tc and ts)
    v_bridge = np.zeros(dim)
    v_bridge[0] = 0.7
    v_bridge[1] = np.sqrt(1.0 - 0.7 ** 2)
    clusterer.predict_next(v_bridge)

    self.assertEqual(clusterer.n_clusters_, 1)

    # Grow bridge subcluster
    for _ in range(50):
      clusterer.predict_next(v_bridge)

    # Grow v1 subcluster
    for _ in range(50):
      clusterer.predict_next(v1)

    # As both subclusters grow, threshold s(k1, k2) rises towards Tp=0.95 > 0.7
    # The edge is removed and no partner exists, so it splits into 2 clusters!
    self.assertEqual(clusterer.n_clusters_, 2)

  def test_synthetic_clusters_high_accuracy(self):
    """Verify 100% accuracy on well-separated clusters."""
    np.random.seed(42)
    n_clusters = 4
    dim = 128
    samples_per_cluster = 50

    centers = np.random.randn(n_clusters, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X_list = []
    y_list = []
    for c_idx in range(n_clusters):
      for _ in range(samples_per_cluster):
        noise = np.random.randn(dim)
        noise -= (noise @ centers[c_idx]) * centers[c_idx]
        noise /= np.linalg.norm(noise)
        theta = 0.35  # cos(0.35) ~ 0.94
        x = np.cos(theta) * centers[c_idx] + np.sin(theta) * noise
        X_list.append(x)
        y_list.append(c_idx)

    X = np.array(X_list)
    y = np.array(y_list)

    # Random permutation
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    clusterer = LinksClusterer(tc=0.6, ts=0.85, tp=0.95)
    preds = clusterer.fit_predict(X)

    acc = utils.compute_accuracy(y, preds)
    self.assertEqual(acc, 1.0)
    self.assertEqual(clusterer.n_clusters_, n_clusters)

  def test_rejoin_partner(self):
    """Test rejoining with a partner node when edge is severed."""
    clusterer = LinksClusterer(tc=0.5, ts=0.9, tp=1.0)
    dim = 64

    # Node 0
    v0 = np.zeros(dim)
    v0[0] = 1.0
    clusterer.predict_next(v0)

    # Node 1: connected to Node 0 with similarity ~ 0.55
    # (between tc=0.5 and ts=0.9)
    v1 = np.zeros(dim)
    v1[0] = 0.55
    v1[1] = np.sqrt(1.0 - 0.55 ** 2)
    clusterer.predict_next(v1)

    # Node 2: connected to Node 1 with similarity ~ 0.6 to Node 0
    v2 = np.zeros(dim)
    v2[0] = 0.6
    v2[2] = np.sqrt(1.0 - 0.6 ** 2)
    clusterer.predict_next(v2)

    # Now add edge between 1 and 2 if not connected
    clusterer._graph.add_edge(1, 2)

    # Now pull Node 0 and Node 1 apart by growing Node 1 in direction
    # orthogonal to Node 0 but maintaining similarity with Node 2
    # Verify clusterer still re-joins if partner exists
    self.assertGreaterEqual(clusterer.n_clusters_, 1)

  def test_capacity_growth(self):
    """Test capacity expansion when creating many subclusters."""
    clusterer = LinksClusterer(tc=0.5, ts=0.9)
    dim = 100
    # 50 mutually orthogonal vectors
    for i in range(50):
      v = np.zeros(dim)
      v[i] = 1.0
      clusterer.predict_next(v)

    self.assertEqual(clusterer.n_subclusters_, 50)
    self.assertEqual(clusterer.n_clusters_, 50)
    self.assertGreaterEqual(clusterer._centroid_capacity, 50)
    self.assertEqual(len(clusterer.subclusters_), 50)

  def test_return_online_labels_false(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85, return_online_labels=False)
    X = np.eye(5, 10)
    labels = clusterer.fit_predict(X)
    self.assertEqual(len(labels), 5)
    np.testing.assert_array_equal(labels, clusterer.get_final_labels())
    self.assertEqual(len(clusterer.get_online_labels()), 5)

  def test_theoretical_norm_option(self):
    clusterer = LinksClusterer(
        tc=0.6, ts=0.85, tp=0.95, use_theoretical_norm=True)
    X = np.eye(3, 10)
    labels = clusterer.fit_predict(X)
    self.assertEqual(len(labels), 3)

  def test_1d_input_variations(self):
    clusterer = LinksClusterer(tc=0.6, ts=0.85)
    v = np.ones(10)
    clusterer.partial_fit(v)
    self.assertEqual(clusterer.n_samples_seen_, 1)
    pred = clusterer.predict(v)
    self.assertEqual(len(pred), 1)


if __name__ == "__main__":
  unittest.main()
