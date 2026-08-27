"""Links: A High-Dimensional Online Clustering Method."""

import collections
import numpy as np
import typing
from linkscluster import graph
from linkscluster import subcluster
from linkscluster import utils


class LinksClusterer:
  """Links online clustering algorithm for high-dimensional unit vectors."""

  def __init__(
      self,
      cluster_similarity_threshold: typing.Optional[float] = None,
      subcluster_similarity_threshold: typing.Optional[float] = None,
      pair_similarity_maximum: typing.Optional[float] = None,
      tc: typing.Optional[float] = None,
      ts: typing.Optional[float] = None,
      tp: typing.Optional[float] = None,
      use_theoretical_norm: bool = False,
      normalize_input: bool = True,
      return_online_labels: bool = True,
      unassigned_label: int = -1):
    """Initialize the LinksClusterer.

    Args:
      cluster_similarity_threshold: Tc (cos theta_c), similarity threshold for
        determining whether vectors/subclusters belong to the same cluster.
        Must be in (0, 1). Defaults to 0.5 if neither this nor tc is given.
      subcluster_similarity_threshold: Ts, threshold for grouping vectors into
        the same fine-grained subcluster. Must be in (0, 1) and typically
        Ts >= Tc. Defaults to 0.8 if neither this nor ts is given.
      pair_similarity_maximum: Tp, upper limit for pair similarity taking into
        account intra-subcluster correlations and anisotropy. Must be in
        (Tc^2, 1]. Defaults to 1.0 (isotropic) if not specified.
      tc: alias for cluster_similarity_threshold (Tc in paper).
      ts: alias for subcluster_similarity_threshold (Ts in paper).
      tp: alias for pair_similarity_maximum (Tp in paper).
      use_theoretical_norm: if True, centroid normalization uses Eq. 12
        theoretical norm \\sqrt{k^2 cos^2\\theta_c + k sin^2\\theta_c}; if False
        (recommended for arbitrary empirical distributions), normalizes by
        empirical L2 norm to ensure unit length on S^{N-1}.
      normalize_input: if True, input vectors are L2-normalized to unit length.
      return_online_labels: if True, fit_predict returns the online assigned
        cluster IDs at vector arrival time. If False, returns final cluster IDs
        reflecting all subsequent splits and merges.
      unassigned_label: integer label assigned to out-of-sample vectors in
        predict() that do not meet the cluster membership threshold.
    """
    effective_tc = tc if tc is not None else cluster_similarity_threshold
    effective_ts = ts if ts is not None else subcluster_similarity_threshold
    effective_tp = tp if tp is not None else pair_similarity_maximum

    self.tc = 0.5 if effective_tc is None else float(effective_tc)
    self.ts = 0.8 if effective_ts is None else float(effective_ts)
    self.tp = 1.0 if effective_tp is None else float(effective_tp)

    if not (0.0 < self.tc < 1.0):
      raise ValueError(f"tc must be in (0, 1), got {self.tc}")
    if not (0.0 < self.ts < 1.0):
      raise ValueError(f"ts must be in (0, 1), got {self.ts}")

    self._tc_sq = self.tc ** 2
    if self.tp < self._tc_sq or self.tp > 1.0:
      raise ValueError(
          f"tp must be in [{self._tc_sq:.4f}, 1.0], got {self.tp}")

    # Aliases matching paper notation
    self.cluster_similarity_threshold = self.tc
    self.subcluster_similarity_threshold = self.ts
    self.pair_similarity_maximum = self.tp
    self.Tc = self.tc
    self.Ts = self.ts
    self.Tp = self.tp

    self.use_theoretical_norm = use_theoretical_norm
    self.normalize_input = normalize_input
    self.return_online_labels = return_online_labels
    self.unassigned_label = unassigned_label

    self._alpha = (1.0 / self._tc_sq) - 1.0
    if self.tp < 1.0:
      self._beta = (self.tp - self._tc_sq) / (1.0 - self._tc_sq)
    else:
      self._beta = 1.0

    self.reset()

  def reset(self):
    """Reset the clusterer state to initial empty condition."""
    self._subclusters: typing.List[subcluster.Subcluster] = []
    self._subcluster_map: typing.Dict[int, subcluster.Subcluster] = {}
    self._centroids: np.ndarray = np.empty((0, 0), dtype=np.float64)
    self._centroid_capacity: int = 0

    self._graph = graph.SubclusterGraph()
    self._cluster_subclusters: typing.Dict[int, typing.Set[int]] = (
        collections.defaultdict(set))

    self._next_sc_id: int = 0
    self._next_cluster_id_val: int = 0
    self.n_samples_seen_: int = 0
    self.n_features_in_: typing.Optional[int] = None

    self._online_labels: typing.List[int] = []
    self.labels_: typing.Optional[np.ndarray] = None
    self.online_labels_: typing.Optional[np.ndarray] = None
    self.final_labels_: typing.Optional[np.ndarray] = None

  @property
  def n_clusters_(self) -> int:
    """Number of active clusters (connected components in the graph)."""
    return len(self._cluster_subclusters)

  @property
  def n_subclusters_(self) -> int:
    """Number of active subclusters."""
    return len(self._subclusters)

  @property
  def subclusters_(self) -> typing.List[subcluster.Subcluster]:
    """List of active subclusters."""
    return list(self._subclusters)

  def _alloc_cluster_id(self) -> int:
    """Allocate a new unique integer cluster ID."""
    cid = self._next_cluster_id_val
    self._next_cluster_id_val += 1
    return cid

  def s(self, k: int, k_prime: int = 1) -> float:
    """Cosine similarity threshold s(k) or s(k, k') (Eqs. 13 & 16 in paper).

    Args:
      k: size of the first subcluster
      k_prime: size of the second subcluster (defaults to 1 for s(k))

    Returns:
      threshold: float threshold value
    """
    term1 = 1.0 + self._alpha / k
    term2 = 1.0 + self._alpha / k_prime
    return float(1.0 / np.sqrt(term1 * term2))

  def s_tilde(self, k: int, k_prime: int = 1) -> float:
    """Anisotropic threshold \\tilde{s}(k, k') or \\tilde{s}(k) (Eqs. 24 & 25).

    Args:
      k: size of the first subcluster
      k_prime: size of the second subcluster (defaults to 1 for \\tilde{s}(k))

    Returns:
      threshold: float interpolated threshold value
    """
    raw_s = self.s(k, k_prime)
    if self._beta == 1.0:
      return raw_s
    return float(self._tc_sq + self._beta * (raw_s - self._tc_sq))

  def _ensure_centroid_capacity(self, needed: int, dim: int):
    """Ensure contiguous centroid matrix has enough capacity."""
    if self._centroid_capacity == 0:
      self._centroid_capacity = max(32, needed)
      self._centroids = np.zeros(
          (self._centroid_capacity, dim), dtype=np.float64)
    elif needed > self._centroid_capacity:
      new_cap = max(self._centroid_capacity * 2, needed)
      new_centroids = np.zeros((new_cap, dim), dtype=np.float64)
      n_active = len(self._subclusters)
      new_centroids[:n_active] = self._centroids[:n_active]
      self._centroids = new_centroids
      self._centroid_capacity = new_cap

  def _add_subcluster(
      self,
      vector: np.ndarray,
      vector_idx: int,
      cluster_id: int) -> int:
    """Create a new subcluster and register it."""
    dim = len(vector)
    idx = len(self._subclusters)
    self._ensure_centroid_capacity(idx + 1, dim)

    sc_id = self._next_sc_id
    self._next_sc_id += 1

    sc = subcluster.Subcluster(
        subcluster_id=sc_id,
        cluster_id=cluster_id,
        vector=vector,
        vector_idx=vector_idx,
        tc=self.tc,
        index=idx,
        use_theoretical_norm=self.use_theoretical_norm)

    self._subclusters.append(sc)
    self._subcluster_map[sc_id] = sc
    self._centroids[idx] = sc.centroid
    self._graph.add_node(sc_id)
    self._cluster_subclusters[cluster_id].add(sc_id)
    return sc_id

  def _remove_subcluster_by_index(self, idx: int):
    """Remove a subcluster at idx using swap-and-pop for O(1) removal."""
    last_idx = len(self._subclusters) - 1
    sc_to_remove = self._subclusters[idx]
    sc_id = sc_to_remove.subcluster_id

    if idx != last_idx:
      last_sc = self._subclusters[last_idx]
      self._subclusters[idx] = last_sc
      last_sc.index = idx
      self._centroids[idx] = self._centroids[last_idx]

    self._subclusters.pop()
    del self._subcluster_map[sc_id]

    cid = sc_to_remove.cluster_id
    if cid in self._cluster_subclusters:
      self._cluster_subclusters[cid].discard(sc_id)
      if not self._cluster_subclusters[cid]:
        del self._cluster_subclusters[cid]

  def predict_next(self, x: np.ndarray) -> int:
    """Assign cluster ID to a new incoming vector in an online stream.

    Implements Section 3.3 and Section 3.4 of the paper.

    Args:
      x: 1D or 2D vector of shape (n_features,) or (1, n_features)

    Returns:
      cluster_id: integer cluster identifier assigned to vector x
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    if self.n_features_in_ is None:
      self.n_features_in_ = len(x)
    elif len(x) != self.n_features_in_:
      raise ValueError(
          f"Expected vector of dimension {self.n_features_in_}, got {len(x)}")

    if self.normalize_input:
      x = utils.l2_normalize(x)

    t = self.n_samples_seen_
    self.n_samples_seen_ += 1

    num_subclusters = len(self._subclusters)
    if num_subclusters == 0:
      # First vector starts first subcluster and cluster
      cid = self._alloc_cluster_id()
      self._add_subcluster(x, t, cid)
      self._online_labels.append(cid)
      return cid

    # Eq. 19: J = argmax_j { x . \\hat{\\mu}_j }
    sims = self._centroids[:num_subclusters] @ x
    J = int(np.argmax(sims))
    x_dot_mu_J = float(sims[J])
    sc_J = self._subclusters[J]
    k_J = sc_J.k

    # Eq. 20: x . \\hat{\\mu}_J >= Ts
    if x_dot_mu_J >= self.Ts:
      # Inequality 20 holds: add x to subcluster J
      assigned_cid = sc_J.cluster_id
      self._online_labels.append(assigned_cid)
      sc_J.add_vector(x, t)
      self._centroids[J] = sc_J.centroid

      # Updating clusters (Section 3.4)
      self._update_clusters(sc_J.subcluster_id)
      return assigned_cid
    else:
      # Inequality 20 does not hold: start a new subcluster containing just x.
      # Eq. 21: x . \\hat{\\mu}_J >= s(k_J) (anisotropic: \\tilde{s}(k_J))
      thresh_kJ = self.s_tilde(k_J, 1)
      if x_dot_mu_J >= thresh_kJ:
        # Include new subcluster in same cluster as J, add edge (new_sc, J)
        assigned_cid = sc_J.cluster_id
        new_sc_id = self._add_subcluster(x, t, assigned_cid)
        self._graph.add_edge(new_sc_id, sc_J.subcluster_id)
      else:
        # Start new cluster
        assigned_cid = self._alloc_cluster_id()
        self._add_subcluster(x, t, assigned_cid)

      self._online_labels.append(assigned_cid)
      return assigned_cid

  def update(self, x: np.ndarray) -> int:
    """Alias for predict_next."""
    return self.predict_next(x)

  def _update_clusters(self, start_sc_id: int):
    """Perform subcluster merging and edge validity checks (Section 3.4).

    Args:
      start_sc_id: identifier of the updated subcluster i
    """
    # 1. Subcluster merging:
    # "If this brings it within the subcluster similarity threshold of the
    # centroid of another subcluster currently joined to the first by an edge,
    # then the two are merged. In other words, if \\hat{\\mu}_i . \\hat{\\mu}_j
    # >= Ts, then nodes i and j are replaced with a single node containing the
    # vectors of both, and with the edge connections of both. Since the merging
    # process also results in a new subcluster centroid, this check is
    # continued recursively on affected subclusters."
    i = start_sc_id
    while i in self._subcluster_map:
      sc_i = self._subcluster_map[i]
      merged_any = False
      nbr_ids = list(self._graph.neighbors(i))

      for j in nbr_ids:
        if j not in self._subcluster_map:
          continue
        sc_j = self._subcluster_map[j]
        # \\hat{\\mu}_i . \\hat{\\mu}_j >= Ts
        mu_i_dot_mu_j = float(np.dot(sc_i.mu_hat, sc_j.mu_hat))
        if mu_i_dot_mu_j >= self.Ts:
          # Merge node j into node i
          sc_i.merge_with(sc_j)
          self._centroids[sc_i.index] = sc_i.centroid
          self._graph.merge_nodes(keep_id=i, remove_id=j)
          self._remove_subcluster_by_index(sc_j.index)
          merged_any = True
          break  # Check continued recursively on affected subcluster i

      if not merged_any:
        break

    if i not in self._subcluster_map:
      return

    # 2. Edge validity check:
    # "Next, the edges joining affected nodes are checked for validity.
    # The edge joining subclusters i and j is removed if the following does
    # not continue to hold:
    # \\hat{\\mu}_i . \\hat{\\mu}_j >= s(k_i, k_j) (Eq. 22)"
    sc_i = self._subcluster_map[i]
    nbr_ids = list(self._graph.neighbors(i))

    for j in nbr_ids:
      if not self._graph.has_edge(i, j):
        continue
      if j not in self._subcluster_map:
        continue
      sc_j = self._subcluster_map[j]
      k_i = sc_i.k
      k_j = sc_j.k
      mu_i_dot_mu_j = float(np.dot(sc_i.mu_hat, sc_j.mu_hat))
      thresh_ij = self.s_tilde(k_i, k_j)

      if mu_i_dot_mu_j < thresh_ij:
        # Edge (i, j) removed
        self._graph.remove_edge(i, j)
        cluster_nodes = self._cluster_subclusters.get(sc_i.cluster_id, set())
        severed, comp_j = self._graph.check_severed(
            start_id=j, target_id=i, allowed_nodes=cluster_nodes)

        if severed:
          # "After severing a cluster in two by removing an edge, an attempt is
          # made to re-join the two parts by adding an edge from the affected
          # node to a new partner node that does satisfy inequality 22."
          best_partner = None
          best_partner_sim = -2.0
          for w in comp_j:
            sc_w = self._subcluster_map[w]
            k_w = sc_w.k
            mu_i_dot_mu_w = float(np.dot(sc_i.mu_hat, sc_w.mu_hat))
            thresh_iw = self.s_tilde(k_i, k_w)
            if mu_i_dot_mu_w >= thresh_iw and mu_i_dot_mu_w > best_partner_sim:
              best_partner = w
              best_partner_sim = mu_i_dot_mu_w

          if best_partner is not None:
            # Re-join parts by adding edge (i, best_partner)
            self._graph.add_edge(i, best_partner)
          else:
            # "If no such partner is found, then the cluster remains permanently
            # split."
            new_cid = self._alloc_cluster_id()
            old_cid = sc_i.cluster_id
            for w in comp_j:
              sc_w = self._subcluster_map[w]
              self._cluster_subclusters[old_cid].discard(w)
              self._cluster_subclusters[new_cid].add(w)
              sc_w.cluster_id = new_cid
            if not self._cluster_subclusters[old_cid]:
              del self._cluster_subclusters[old_cid]

  def partial_fit(self, X: np.ndarray, y=None) -> "LinksClusterer":
    """Incrementally cluster a batch of vectors.

    Args:
      X: numpy array of shape (n_samples, n_features) or (n_features,)
      y: ignored

    Returns:
      self
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
      X = np.expand_dims(X, 0)
    for vec in X:
      self.predict_next(vec)
    return self

  def predict_stream(
      self,
      stream: typing.Iterable[np.ndarray]) -> typing.Iterator[int]:
    """Stream vectors one by one and yield predicted cluster IDs."""
    for vec in stream:
      yield self.predict_next(vec)

  def fit(self, X: np.ndarray, y=None) -> "LinksClusterer":
    """Fit the Links clusterer on a dataset.

    Args:
      X: numpy array of shape (n_samples, n_features)
      y: ignored

    Returns:
      self
    """
    self.fit_predict(X)
    return self

  def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
    """Fit the clusterer and return cluster labels for X.

    Args:
      X: numpy array of shape (n_samples, n_features)
      y: ignored

    Returns:
      labels: numpy array of shape (n_samples,)
    """
    self.reset()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
      raise ValueError(
          f"X must be 2D array of shape (n_samples, n_features), got {X.shape}")
    if X.shape[0] == 0:
      self.labels_ = np.empty((0,), dtype=int)
      self.online_labels_ = np.empty((0,), dtype=int)
      self.final_labels_ = np.empty((0,), dtype=int)
      return self.labels_

    for vec in X:
      self.predict_next(vec)

    self.online_labels_ = np.array(self._online_labels, dtype=int)
    self.final_labels_ = self.get_final_labels()

    if self.return_online_labels:
      self.labels_ = self.online_labels_.copy()
    else:
      self.labels_ = self.final_labels_.copy()

    return self.labels_

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict cluster IDs for X.

    If the model has not been fitted, calls fit_predict(X). If already fitted,
    assigns each vector to its most similar subcluster's cluster without
    modifying clusterer state.

    Args:
      X: numpy array of shape (n_samples, n_features) or (n_features,)

    Returns:
      labels: numpy array of shape (n_samples,)
    """
    if self.n_samples_seen_ == 0:
      return self.fit_predict(X)

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
      X = np.expand_dims(X, 0)
    if self.normalize_input:
      X = utils.l2_normalize(X)

    num_subclusters = len(self._subclusters)
    if num_subclusters == 0:
      return np.full(X.shape[0], self.unassigned_label, dtype=int)

    sims = X @ self._centroids[:num_subclusters].T
    best_j = np.argmax(sims, axis=1)
    preds = np.empty(X.shape[0], dtype=int)

    for idx in range(X.shape[0]):
      j = int(best_j[idx])
      sim = float(sims[idx, j])
      sc = self._subclusters[j]
      thresh = self.s_tilde(sc.k, 1)
      if sim >= thresh or self.unassigned_label is None:
        preds[idx] = sc.cluster_id
      else:
        preds[idx] = self.unassigned_label

    return preds

  def get_online_labels(self) -> np.ndarray:
    """Return the online cluster IDs assigned at vector arrival times."""
    return np.array(self._online_labels, dtype=int)

  def get_final_labels(self) -> np.ndarray:
    """Return post-hoc cluster IDs reflecting all subsequent splits and merges.
    """
    labels = np.zeros(self.n_samples_seen_, dtype=int)
    for sc in self._subclusters:
      for v_idx in sc.vector_indices:
        labels[v_idx] = sc.cluster_id
    return labels
