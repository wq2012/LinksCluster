"""Utility functions for Links clustering."""

import numpy as np
import scipy.optimize
import typing


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  """Normalize vectors to unit L2 length.

  Args:
    vectors: numpy array of shape (n_samples, n_features) or (n_features,)
    eps: small epsilon to avoid division by zero

  Returns:
    normalized_vectors: numpy array of same shape with unit L2 norm
  """
  if vectors.ndim == 1:
    norm = np.linalg.norm(vectors)
    if norm > eps:
      return vectors / norm
    return vectors.copy()
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  norms = np.maximum(norms, eps)
  return vectors / norms


def single_threshold(k: int, tc: float) -> float:
  """Compute threshold s(k) for single subcluster (Equation 13 in paper).

  Args:
    k: number of vectors in the subcluster
    tc: cluster similarity threshold (cos theta_c), 0 < tc < 1

  Returns:
    threshold: float cosine similarity threshold
  """
  if k < 1:
    raise ValueError("k must be >= 1")
  if not (0.0 < tc < 1.0):
    raise ValueError("tc must be in (0, 1)")
  tc_sq = tc ** 2
  denom = np.sqrt((1.0 / k) + (1.0 - 1.0 / k) * tc_sq)
  return float(tc_sq / denom)


def multi_threshold(k1: int, k2: int, tc: float) -> float:
  """Compute threshold s(k1, k2) for subcluster pair (Equation 16 in paper).

  Args:
    k1: number of vectors in the first subcluster
    k2: number of vectors in the second subcluster
    tc: cluster similarity threshold (cos theta_c), 0 < tc < 1

  Returns:
    threshold: float cosine similarity threshold
  """
  if k1 < 1 or k2 < 1:
    raise ValueError("k1 and k2 must be >= 1")
  if not (0.0 < tc < 1.0):
    raise ValueError("tc must be in (0, 1)")
  tc_sq = tc ** 2
  alpha = (1.0 / tc_sq) - 1.0
  term1 = 1.0 + alpha / k1
  term2 = 1.0 + alpha / k2
  return float(1.0 / np.sqrt(term1 * term2))


def anisotropic_threshold(
    k1: int,
    k2: int,
    tc: float,
    tp: typing.Optional[float] = None) -> float:
  """Compute anisotropic threshold s_tilde(k1, k2) (Equation 24 in paper).

  Args:
    k1: number of vectors in first subcluster
    k2: number of vectors in second subcluster
    tc: cluster similarity threshold (cos theta_c), 0 < tc < 1
    tp: pair similarity maximum, tc^2 < tp <= 1. If None or 1.0, isotropic.

  Returns:
    threshold: float cosine similarity threshold
  """
  raw_s = multi_threshold(k1, k2, tc)
  if tp is None or tp >= 1.0:
    return raw_s
  tc_sq = tc ** 2
  if tp < tc_sq:
    raise ValueError(f"tp must be >= tc^2 ({tc_sq}), got {tp}")
  beta = (tp - tc_sq) / (1.0 - tc_sq)
  return float(tc_sq + beta * (raw_s - tc_sq))


def enforce_ordered_labels(labels: np.ndarray) -> np.ndarray:
  """Transform the label sequence to an ordered form starting at 0.

  Args:
    labels: array of integer cluster IDs

  Returns:
    ordered_labels: array of integers starting at 0 in order of appearance
  """
  labels = np.asarray(labels)
  new_labels = labels.copy()
  max_label = -1
  label_map: typing.Dict[typing.Any, int] = {}
  for element in labels.tolist():
    if element not in label_map:
      max_label += 1
      label_map[element] = max_label
  for key, val in label_map.items():
    new_labels[labels == key] = val
  return new_labels


def compute_accuracy(
    true_labels: np.ndarray,
    pred_labels: np.ndarray) -> float:
  """Compute clustering accuracy using Hungarian matching (Section 3.6).

  Bijectively maps predicted clusters to ground truth clusters to maximize
  accuracy.

  Args:
    true_labels: ground truth labels of shape (n_samples,)
    pred_labels: predicted cluster labels of shape (n_samples,)

  Returns:
    accuracy: float fraction of correct assignments in [0, 1]
  """
  true_labels = np.asarray(true_labels)
  pred_labels = np.asarray(pred_labels)
  if true_labels.shape != pred_labels.shape:
    raise ValueError("true_labels and pred_labels must have the same shape")
  if len(true_labels) == 0:
    return 1.0

  _, true_idx = np.unique(true_labels, return_inverse=True)
  _, pred_idx = np.unique(pred_labels, return_inverse=True)
  n_true = int(np.max(true_idx) + 1)
  n_pred = int(np.max(pred_idx) + 1)

  contingency = np.zeros((n_true, n_pred), dtype=np.int64)
  np.add.at(contingency, (true_idx, pred_idx), 1)

  row_ind, col_ind = scipy.optimize.linear_sum_assignment(-contingency)
  matched_count = contingency[row_ind, col_ind].sum()
  return float(matched_count) / len(true_labels)
