"""Example configurations for Links online clustering."""

from linkscluster import links_clusterer

LinksClusterer = links_clusterer.LinksClusterer

# Default configuration for general high-dimensional unit embeddings
default_links_clusterer = LinksClusterer(
    cluster_similarity_threshold=0.5,
    subcluster_similarity_threshold=0.8,
    pair_similarity_maximum=1.0)

# Configuration tuned for 128-dimensional FaceNet CNN face embeddings
# (Schroff et al., CVPR 2015)
facenet_clusterer = LinksClusterer(
    cluster_similarity_threshold=0.6,
    subcluster_similarity_threshold=0.85,
    pair_similarity_maximum=0.95)

# Configuration tuned for 256-dimensional LSTM voice embeddings
# (Wan et al., 2017 GE2E loss; Wang et al., 2017 Speaker Diarization with LSTM)
ge2e_voice_clusterer = LinksClusterer(
    cluster_similarity_threshold=0.55,
    subcluster_similarity_threshold=0.8,
    pair_similarity_maximum=0.9)
