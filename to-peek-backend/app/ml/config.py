"""
Configuration and constants for topic modeling.
"""

# =============================================================================
# EMBEDDING MODEL
# =============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================

# UMAP: Dimensionality reduction before clustering
UMAP_N_NEIGHBORS = 15        # Higher = more global structure, lower = more local
UMAP_N_COMPONENTS = 5        # Number of dimensions to reduce to
UMAP_MIN_DIST = 0.0          # 0.0 = tight clusters, 1.0 = spread out
UMAP_METRIC = "cosine"       # Distance metric (cosine works well for text)

# HDBSCAN: Clustering algorithm (base values, adapted dynamically)
# Target: ~30 topics level-1, <20% outliers
HDBSCAN_MIN_CLUSTER_SIZE_BASE = 15   # Minimum base value (lower = more small clusters allowed)
HDBSCAN_MIN_CLUSTER_SIZE_RATIO = 70  # n_docs / ratio = min_cluster_size
HDBSCAN_MIN_SAMPLES_BASE = 3         # Minimum base value (lower = less strict density)
HDBSCAN_MIN_SAMPLES_RATIO = 500      # n_docs / ratio = min_samples (higher = less outliers)
HDBSCAN_CLUSTER_EPSILON = 0.0        # Distance threshold (0 = auto)
HDBSCAN_METRIC = "euclidean"         # Distance metric for clustering

# BERTopic
MIN_TOPIC_SIZE = 3           # Minimum documents per topic (lower = keep small clusters)

# Sub-clustering (for splitting low-quality clusters)
SUB_MIN_CLUSTER_SIZE_RATIO = 10   # parent_size / ratio = min_cluster_size
SUB_MIN_CLUSTER_SIZE_BASE = 5     # Minimum base value
SUB_MIN_TOPIC_SIZE_RATIO = 15     # parent_size / ratio = min_topic_size
SUB_MIN_TOPIC_SIZE_BASE = 5       # Minimum base value


def get_adaptive_hdbscan_params(n_docs: int) -> dict:
    """
    Calculate adaptive HDBSCAN parameters based on corpus size.
    Target: ~30 topics, <20% outliers
    
    Examples:
        - 500 docs  → min_cluster_size=15, min_samples=3
        - 2000 docs → min_cluster_size=28, min_samples=4
        - 10000 docs → min_cluster_size=142, min_samples=20
    """
    min_cluster_size = max(HDBSCAN_MIN_CLUSTER_SIZE_BASE, n_docs // HDBSCAN_MIN_CLUSTER_SIZE_RATIO)
    min_samples = max(HDBSCAN_MIN_SAMPLES_BASE, n_docs // HDBSCAN_MIN_SAMPLES_RATIO)
    
    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }


def get_adaptive_sub_params(parent_size: int) -> dict:
    """
    Calculate adaptive sub-clustering parameters based on parent cluster size.
    
    Examples:
        - 50 docs  → min_cluster_size=5, min_topic_size=5
        - 300 docs → min_cluster_size=30, min_topic_size=20
    """
    return {
        "min_cluster_size": max(SUB_MIN_CLUSTER_SIZE_BASE, parent_size // SUB_MIN_CLUSTER_SIZE_RATIO),
        "min_topic_size": max(SUB_MIN_TOPIC_SIZE_BASE, parent_size // SUB_MIN_TOPIC_SIZE_RATIO),
    }
NR_TOPICS = None             # None = auto, or set a number to force reduction

# Vectorizer
VECTORIZER_MIN_DF = 2        # Word must appear in at least N docs
VECTORIZER_NGRAM_RANGE = (1, 2)  # (1,1)=unigrams, (1,2)=uni+bigrams

# Semantic word extraction
SEMANTIC_TOP_N_WORDS = 10    # Number of semantic words to extract per topic
SEMANTIC_VOCAB_MIN_DF = 5    # Word must appear in at least N docs to be in vocab
SEMANTIC_NGRAM_RANGE = (1, 3)  # Include unigrams, bigrams, trigrams
SEMANTIC_CANDIDATES = 100     # Pre-filter top N candidates before MMR
MMR_LAMBDA = 0.7             # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity

# Hierarchical extraction
SILHOUETTE_THRESHOLD = 0.15  # Below this → split into sub-topics
MAX_DEPTH = 1                # Max nesting levels

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

