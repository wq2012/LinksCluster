"""Subcluster representation for Links online clustering."""

import numpy as np
import typing


class Subcluster:
  """A subcluster node representing a collection of vectors in Links.

  Corresponds to the two-level hierarchy nodes in Section 3.2 of the paper.
  Maintains vector count k, cumulative sum_vector, and centroid \\hat{\\mu}.
  """

  def __init__(
      self,
      subcluster_id: int,
      cluster_id: int,
      vector: np.ndarray,
      vector_idx: int,
      tc: float,
      index: int,
      use_theoretical_norm: bool = False):
    """Initialize a new subcluster with an initial vector.

    Args:
      subcluster_id: unique integer identifier for this subcluster
      cluster_id: cluster identifier for the connected component
      vector: initial unit-length vector of shape (n_features,)
      vector_idx: index of the vector in the stream / dataset
      tc: cluster similarity threshold Tc = cos(theta_c)
      index: index of this subcluster in the contiguous centroid matrix
      use_theoretical_norm: if True, use Eq. 12 theoretical norm; if False,
        normalize by empirical L2 norm to ensure unit length on S^{N-1}
    """
    self.subcluster_id = subcluster_id
    self.cluster_id = cluster_id
    self.count = 1
    self.sum_vector = np.array(vector, dtype=np.float64, copy=True)
    self.vector_indices: typing.List[int] = [vector_idx]
    self.tc = tc
    self.index = index
    self.use_theoretical_norm = use_theoretical_norm
    self.centroid = np.zeros_like(self.sum_vector)
    self._update_centroid()

  @property
  def id(self) -> int:
    """Subcluster ID."""
    return self.subcluster_id

  @property
  def k(self) -> int:
    """Number of vectors in this subcluster (k in paper)."""
    return self.count

  @property
  def mu_hat(self) -> np.ndarray:
    """Centroid vector \\hat{\\mu} of this subcluster (Eq. 12 in paper)."""
    return self.centroid

  def _update_centroid(self):
    """Update centroid \\hat{\\mu} based on sum_vector and count k (Eq. 12)."""
    if self.use_theoretical_norm:
      k = self.count
      tc_sq = self.tc ** 2
      # Eq. 12: \\hat{\\mu} = 1 / \\sqrt{k^2 cos^2\\theta_c + k sin^2\\theta_c}
      #         * \\sum x_i
      denom = np.sqrt(k ** 2 * tc_sq + k * (1.0 - tc_sq))
      if denom > 1e-12:
        self.centroid = self.sum_vector / denom
      else:
        self.centroid = self.sum_vector.copy()
    else:
      # Empirical unit-norm centroid on S^{N-1}
      norm = np.linalg.norm(self.sum_vector)
      if norm > 1e-12:
        self.centroid = self.sum_vector / norm
      else:
        self.centroid = self.sum_vector.copy()

  def add_vector(self, vector: np.ndarray, vector_idx: int):
    """Add a new vector to this subcluster and recompute centroid.

    Args:
      vector: vector of shape (n_features,)
      vector_idx: index of the vector
    """
    self.count += 1
    self.sum_vector += vector
    self.vector_indices.append(vector_idx)
    self._update_centroid()

  def merge_with(self, other: "Subcluster"):
    """Merge another subcluster into this subcluster.

    Args:
      other: the other Subcluster instance to merge into self
    """
    self.count += other.count
    self.sum_vector += other.sum_vector
    self.vector_indices.extend(other.vector_indices)
    self._update_centroid()
