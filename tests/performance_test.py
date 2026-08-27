"""Performance and efficiency tests for LinksClusterer."""

import time
import unittest
import numpy as np
from linkscluster import links_clusterer

LinksClusterer = links_clusterer.LinksClusterer


class TestPerformance(unittest.TestCase):
  """Performance and throughput tests."""

  def test_1000_samples_throughput(self):
    np.random.seed(42)
    n_samples = 1000
    dim = 128
    # 5 clusters
    centers = np.random.randn(5, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = []
    for i in range(n_samples):
      c = centers[i % 5]
      noise = np.random.randn(dim) * 0.1
      v = c + noise
      v /= np.linalg.norm(v)
      X.append(v)
    X = np.array(X)

    clusterer = LinksClusterer(tc=0.6, ts=0.85, tp=0.95)
    t0 = time.perf_counter()
    clusterer.fit_predict(X)
    elapsed = time.perf_counter() - t0

    throughput = n_samples / elapsed
    print(f"\nThroughput: {throughput:.0f} samples/sec ({elapsed*1000:.2f} ms)")
    # Assert fast throughput (at least 5,000 samples/sec)
    self.assertGreater(throughput, 5000)

  def test_5000_samples_throughput(self):
    np.random.seed(42)
    n_samples = 5000
    dim = 128
    centers = np.random.randn(10, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = []
    for i in range(n_samples):
      c = centers[i % 10]
      noise = np.random.randn(dim) * 0.08
      v = c + noise
      v /= np.linalg.norm(v)
      X.append(v)
    X = np.array(X)

    clusterer = LinksClusterer(tc=0.6, ts=0.85, tp=0.95)
    t0 = time.perf_counter()
    clusterer.fit_predict(X)
    elapsed = time.perf_counter() - t0

    throughput = n_samples / elapsed
    print(
        f"5000 samples Throughput: {throughput:.0f} samples/sec "
        f"({elapsed*1000:.2f} ms)")
    self.assertGreater(throughput, 5000)


if __name__ == "__main__":
  unittest.main()
