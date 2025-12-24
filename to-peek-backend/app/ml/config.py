"""
Configuration and constants for topic modeling.
"""

# =============================================================================
# EMBEDDING MODEL
# =============================================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# PCA: Fast pre-reduction before UMAP (hybrid approach)
PCA_N_COMPONENTS = 50        # Reduce 384D -> 50D (keeps ~95% variance, 7x faster UMAP)

# UMAP: Dimensionality reduction before clustering (operates on PCA-reduced data)
UMAP_N_NEIGHBORS = 15        # Higher = more global structure, lower = more local
UMAP_N_COMPONENTS = 5        # Number of dimensions to reduce to
UMAP_MIN_DIST = 0.0          # 0.0 = tight clusters, 1.0 = spread out
UMAP_METRIC = "cosine"       # Distance metric (cosine works well for text)

# UMAP: 2D projection for visualization (separate from clustering)
VIZ_UMAP_N_NEIGHBORS = 15    # Balance local/global structure
VIZ_UMAP_MIN_DIST = 0.2      # Slight spread for readability
VIZ_UMAP_METRIC = "cosine"   # Same as clustering

# =============================================================================
# HDBSCAN: Permissive clustering (many topics, few outliers)
# =============================================================================
# Strategy: Accept many micro-topics initially, then reduce with BERTopic
HDBSCAN_MIN_CLUSTER_SIZE = 10        # Low = more small clusters allowed
HDBSCAN_MIN_SAMPLES = 3              # Low = less strict density requirement
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"  # More stable than 'leaf'
HDBSCAN_CLUSTER_EPSILON = 0.05       # Merge very close clusters
HDBSCAN_METRIC = "euclidean"         # Distance metric for clustering

# =============================================================================
# BERTopic: Topic reduction and outlier handling
# =============================================================================
MIN_TOPIC_SIZE = 5                   # Accept small topics initially
TARGET_TOPICS_LEVEL_0 = 15           # reduce_topics() target (10-20 range)
OUTLIER_REDUCTION_STRATEGY = "embeddings"  # "embeddings", "probabilities", or "c-tf-idf"
OUTLIER_REDUCTION_THRESHOLD = 0.5    # Distance threshold for reassignment

# =============================================================================
# Hierarchy: Mean-distance-based splitting
# =============================================================================
MEAN_DISTANCE_THRESHOLD = 0.75       # High distance = dispersed cluster -> split
MAX_DEPTH = 1                        # Max nesting levels

# Sub-clustering (for splitting low-quality clusters)
SUB_MIN_CLUSTER_SIZE_RATIO = 10      # parent_size / ratio = min_cluster_size
SUB_MIN_CLUSTER_SIZE_BASE = 5        # Minimum base value
SUB_MIN_CLUSTER_SIZE_MAX = 50        # Maximum value (cap for large clusters)
SUB_MIN_TOPIC_SIZE_RATIO = 15        # parent_size / ratio = min_topic_size
SUB_MIN_TOPIC_SIZE_BASE = 5          # Minimum base value
SUB_MIN_TOPIC_SIZE_MAX = 30          # Maximum value (cap for large clusters)


def get_adaptive_sub_params(parent_size: int) -> dict:
    """
    Calculate adaptive sub-clustering parameters based on parent cluster size.
    
    Capped to avoid too-strict params on large clusters.
    
    Examples:
        - 50 docs   → min_cluster_size=5, min_topic_size=5
        - 300 docs  → min_cluster_size=30, min_topic_size=20
        - 2500 docs → min_cluster_size=50 (capped), min_topic_size=30 (capped)
    """
    min_cluster = parent_size // SUB_MIN_CLUSTER_SIZE_RATIO
    min_cluster = max(SUB_MIN_CLUSTER_SIZE_BASE, min(SUB_MIN_CLUSTER_SIZE_MAX, min_cluster))
    
    min_topic = parent_size // SUB_MIN_TOPIC_SIZE_RATIO
    min_topic = max(SUB_MIN_TOPIC_SIZE_BASE, min(SUB_MIN_TOPIC_SIZE_MAX, min_topic))
    
    return {
        "min_cluster_size": min_cluster,
        "min_topic_size": min_topic,
    }

# Vectorizer
VECTORIZER_MIN_DF = 2        # Word must appear in at least N docs
VECTORIZER_NGRAM_RANGE = (1, 2)  # (1,1)=unigrams, (1,2)=uni+bigrams

# Semantic word extraction
SEMANTIC_TOP_N_WORDS = 10    # Number of semantic words to extract per topic
SEMANTIC_VOCAB_MIN_DF = 5    # Word must appear in at least N docs to be in vocab
SEMANTIC_NGRAM_RANGE = (1, 3)  # Include unigrams, bigrams, trigrams
SEMANTIC_CANDIDATES = 100     # Pre-filter top N candidates before MMR
MMR_LAMBDA = 0.7             # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity

# Hierarchical extraction (legacy - kept for reference)
# SILHOUETTE_THRESHOLD = 0.15  # Replaced by PERSISTENCE_THRESHOLD
# MAX_DEPTH defined above in Hierarchy section

# =============================================================================
# STOP WORDS
# =============================================================================

STOP_WORDS = {
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "en", "que", "qui",
    "dans", "ce", "il", "ne", "sur", "se", "pas", "plus", "par", "pour", "au", "aux",
    "avec", "son", "sa", "ses", "ou", "mais", "comme", "on", "tout", "nous", "vous",
    "ils", "elle", "elles", "été", "être", "avoir", "fait", "faire", "dit", "dire",
    "cette", "ces", "sont", "ont", "leur", "leurs", "même", "aussi", "bien", "sans",
    "peut", "tous", "après", "ainsi", "donc", "très", "quand", "ça", "si", "où",
    "vraiment", "merci", "oui", "non", "ok", "moi", "toi", "lui", "eux", "y", "a",
    "mon", "ton", "ma", "ta", "mes", "tes", "nos", "vos",
    # English
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "is", "are", "was", "were", "been", "has", "had", "yes", "no", "ok", "thanks",
    "just", "so", "like", "would", "could", "should", "get", "got", "really",
    "very", "more", "much", "also", "too", "then", "than", "now", "how", "what",
    "when", "where", "why", "who", "which", "there", "here", "all", "some", "any",
}

