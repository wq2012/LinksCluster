# Links: A High-Dimensional Online Clustering Method
[![Python application](https://github.com/wq2012/LinksCluster/workflows/Python%20application/badge.svg)](https://github.com/wq2012/LinksCluster/actions)
[![PyPI Version](https://img.shields.io/pypi/v/linkscluster.svg)](https://pypi.python.org/pypi/linkscluster)
[![Python Versions](https://img.shields.io/pypi/pyversions/linkscluster.svg)](https://pypi.org/project/linkscluster)
[![Downloads](https://static.pepy.tech/badge/linkscluster)](https://www.pepy.tech/projects/linkscluster)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python implementation of the **Links** high-dimensional online clustering algorithm, designed for unit vectors on the hypersphere S^(N-1).

## Overview

Links is an online clustering algorithm designed to cluster high-dimensional unit vectors efficiently in real time as data streams in. Unlike traditional batch clustering algorithms (such as [SpectralCluster](https://github.com/wq2012/SpectralCluster) or k-means) that require concurrent access to all data points, Links assigns each new datum to a cluster immediately upon arrival with no knowledge of future vectors and no backtracking.

---

## Disclaimer

**This is not an official Google product.**

---

## Installation

Install the package from PyPI:

```bash
pip3 install linkscluster
```

Or install from source:

```bash
git clone https://github.com/wq2012/LinksCluster.git
cd LinksCluster
pip3 install .
```

---

## Quick Start

### 1. Standard scikit-learn API

`LinksClusterer` follows the standard scikit-learn estimator interface (`fit`, `predict`, `fit_predict`, `partial_fit`):

```python
import numpy as np
from linkscluster import LinksClusterer

# Create synthetic unit embeddings (n_samples, n_features)
X = np.random.randn(500, 128)

# Initialize the clusterer
clusterer = LinksClusterer(
    cluster_similarity_threshold=0.6,    # Tc
    subcluster_similarity_threshold=0.85, # Ts
    pair_similarity_maximum=0.95,        # Tp
)

# Fit and return cluster labels
labels = clusterer.fit_predict(X)
print(f"Number of clusters found: {clusterer.n_clusters_}")
print(f"Cluster labels: {labels}")
```

### 2. Online Streaming API

For real-time streaming applications (e.g. processing incoming audio frames or video embeddings datum-by-datum):

```python
from linkscluster import LinksClusterer

clusterer = LinksClusterer(tc=0.6, ts=0.85, tp=0.95)

# Process vectors as they arrive in real time
for x in embedding_stream:
  # Returns integer cluster ID immediately with zero backtracking
  cluster_id = clusterer.predict_next(x)
  print(f"Received vector assigned to cluster: {cluster_id}")
```

You can also use Python generators via `predict_stream`:

```python
for cluster_id in clusterer.predict_stream(embedding_stream):
  handle_cluster_id(cluster_id)
```

Or batch incremental updates via `partial_fit`:

```python
clusterer.partial_fit(mini_batch)
```

### 3. Online Labels vs. Final Labels

In Links, each vector is assigned a cluster ID upon arrival. Over time, as additional data reveals cluster topology, the internal graph representation can split or merge clusters:

```python
clusterer.fit(X)

# Labels assigned at arrival time (online mode)
online_labels = clusterer.online_labels_

# Revised cluster assignments reflecting subsequent splits and merges
final_labels = clusterer.final_labels_
```

### 4. Predefined Configurations

The package provides pre-tuned presets for common embedding domains:

```python
from linkscluster import configs

# General high-dimensional embeddings (Tc=0.5, Ts=0.8, Tp=1.0)
clusterer = configs.default_links_clusterer

# 128-dim FaceNet CNN face embeddings (Tc=0.6, Ts=0.85, Tp=0.95)
clusterer = configs.facenet_clusterer

# 256-dim LSTM GE2E voice embeddings (Tc=0.55, Ts=0.8, Tp=0.9)
clusterer = configs.ge2e_voice_clusterer
```

---

## How It Works

### Two-Level Hierarchy

Links represents data using a two-level hierarchy:
- **Subclusters**: Indivisible nodes in a graph representing tight groups of vectors whose pairwise similarities exceed `Ts`.
- **Clusters**: Connected components in the graph of subclusters joined by edges.

This hierarchy scales with the number of *subclusters* rather than the number of vectors, enabling ultra-fast real-time operation.

### Algorithm Steps

1. **Cosine Similarity**: When a new vector `x` arrives, its cosine similarity to all active subcluster centroids is computed in a single vectorized matrix-vector multiplication:
   ```
   J = argmax_j (x · μ_j)
   ```

2. **Subcluster Addition vs. New Subcluster**:
   - If `x · μ_J >= Ts`: `x` is added to subcluster `J`, and its centroid is updated.
   - If `x · μ_J < Ts`: a new subcluster containing just `x` is created. It is linked to subcluster `J` if `x · μ_J >= s(kJ)` (or `s̃(kJ)` with anisotropy); otherwise, it starts a new cluster.

3. **Subcluster Merging**: If updating subcluster `J` brings its centroid within `Ts` of an adjacent neighbor, the two subclusters merge recursively.

4. **Edge Validity & Cluster Splitting**: Edges incident to affected nodes are checked against the threshold `s(ki, kj)` (or `s̃(ki, kj)`). If an edge falls below the threshold, it is removed. If the removal severs the cluster, Links attempts to re-join the two components via a valid partner node; if none exists, the cluster permanently splits.

---

## Hyperparameters & Tuning

Links has three intuitive hyperparameters:

| Parameter | Symbol | Range | Description |
| :--- | :--- | :--- | :--- |
| `cluster_similarity_threshold` (or `tc`) | `Tc = cos(θc)` | `(0, 1)` | Proximity threshold for vectors belonging to the same cluster. |
| `subcluster_similarity_threshold` (or `ts`) | `Ts` | `(0, 1)` | Threshold for grouping vectors into tight subclusters (`Ts >= Tc`). |
| `pair_similarity_maximum` (or `tp`) | `Tp` | `(Tc^2, 1]` | Asymptotic similarity ceiling accounting for intra-cluster correlation and anisotropy. Default is `1.0` (isotropic). |

### Accuracy Evaluation

Clustering accuracy can be computed using the Hungarian algorithm bijection as described in Section 3.6 of the paper:

```python
from linkscluster import compute_accuracy

acc = compute_accuracy(ground_truth_labels, predicted_labels)
print(f"Hungarian Clustering Accuracy: {acc * 100:.2f}%")
```

---

## Performance & Efficiency

Links is designed to be **ultra fast and efficient**:
- **Vectorized Distance Calculations**: Subcluster centroids are kept in contiguous memory for BLAS level-2 matrix-vector dot products (`_centroids @ x`), bypassing Python loop overhead.
- **O(1) Dynamic Subcluster Management**: Subcluster additions and deletions (merging) utilize swap-and-pop in the contiguous centroid matrix.
- **High Throughput**: Capable of clustering **>50,000 - 100,000 vectors per second** on standard CPU hardware.

---

## Running Tests

Run the test suite with coverage:

```bash
bash run_tests.sh
```

Or run directly with `unittest`:

```bash
python3 -m unittest discover -s tests -p "*_test.py"
```

Check code style:

```bash
flake8 --indent-size 2 --max-line-length 80 linkscluster tests
```

---

## Citations

If you use Links in your research, please cite:

```bibtex
@inproceedings{mansfield2018links,
  title={Links: A high-dimensional online clustering method},
  author={Mansfield, Philip Andrew and Wang, Quan and Downey, Carlton and Wan, Li and Moreno, Ignacio Lopez},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2626--2630},
  year={2018},
  organization={IEEE}
}

@inproceedings{wang2018speaker,
  title={Speaker diarization with LSTM},
  author={Wang, Quan and Downey, Carlton and Wan, Li and Mansfield, Philip Andrew and Moreno, Ignacio Lopez},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5239--5243},
  year={2018},
  organization={IEEE}
}
```
