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

# HDBSCAN: Clustering algorithm
HDBSCAN_MIN_CLUSTER_SIZE = 50    # Minimum docs per cluster (smaller = more topics)
HDBSCAN_MIN_SAMPLES = 10         # Core points required (higher = denser clusters)
HDBSCAN_CLUSTER_EPSILON = 0.0    # Distance threshold (0 = auto)
HDBSCAN_METRIC = "euclidean"     # Distance metric for clustering

# BERTopic
MIN_TOPIC_SIZE = 5           # Minimum documents per topic
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

