"""
Configuration and constants for topic modeling.
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
MODELING_DIR = Path(__file__).parent.parent
DATASETS_DIR = MODELING_DIR / "datasets"
CACHE_DIR = MODELING_DIR / "cache"
VOCAB_CACHE_FILE = CACHE_DIR / "vocab_embeddings.parquet"
COMMENTS_CACHE_FILE = CACHE_DIR / "comments_embeddings.parquet"

# =============================================================================
# CLUSTERING PARAMETERS - Tweak these for exploration!
# =============================================================================

# UMAP: Dimensionality reduction before clustering
# Choix basé sur sweep_results_20251222_190342.json (Run 30)
# → 6 topics, 38% outliers, silhouette 0.471 (meilleur ratio topics/qualité)
UMAP_N_NEIGHBORS = 15        # Higher = more global structure, lower = more local
UMAP_N_COMPONENTS = 5        # Number of dimensions to reduce to
UMAP_MIN_DIST = 0.0          # 0.0 = tight clusters, 1.0 = spread out
UMAP_METRIC = "cosine"       # Distance metric (cosine works well for text)

# HDBSCAN: Clustering algorithm
HDBSCAN_MIN_CLUSTER_SIZE = 50    # Minimum docs per cluster (smaller = more topics)
HDBSCAN_MIN_SAMPLES = 10         # Core points required (higher = denser clusters)
HDBSCAN_CLUSTER_EPSILON = 0.0    # Distance threshold (0 = auto, higher = merge close clusters)
HDBSCAN_METRIC = "euclidean"     # Distance metric for clustering

# BERTopic
MIN_TOPIC_SIZE = 5           # Minimum documents per topic
NR_TOPICS = None             # None = auto, or set a number to force reduction

# Vectorizer
VECTORIZER_MIN_DF = 2        # Word must appear in at least N docs
VECTORIZER_NGRAM_RANGE = (1, 2)  # (1,1)=unigrams, (1,2)=uni+bigrams

# Semantic word extraction (centroid → vocabulary approach)
SEMANTIC_TOP_N_WORDS = 10    # Number of semantic words to extract per topic
SEMANTIC_VOCAB_MIN_DF = 5    # Word must appear in at least N docs to be in vocab
SEMANTIC_NGRAM_RANGE = (1, 3)  # Include unigrams, bigrams, trigrams
SEMANTIC_CANDIDATES = 100     # Pre-filter top N candidates before MMR (for speed)
MMR_LAMBDA = 0.7             # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =============================================================================
# STOP WORDS
# =============================================================================

# Fallback static stop words (used if no detected cache exists)
STOP_WORDS_FALLBACK = {
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "en", "que", "qui",
    "dans", "ce", "il", "ne", "sur", "se", "pas", "plus", "par", "pour", "au", "aux",
    "avec", "son", "sa", "ses", "ou", "mais", "comme", "on", "tout", "nous", "vous",
    "ils", "elle", "elles", "été", "être", "avoir", "fait", "faire", "dit", "dire",
    "cette", "ces", "sont", "ont", "leur", "leurs", "même", "aussi", "bien", "sans",
    "peut", "tous", "après", "ainsi", "donc", "très", "quand", "ça", "si", "où",
    "vraiment", "merci", "oui", "non", "ok",
    # English
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "is", "are", "was", "were", "been", "has", "had", "yes", "no", "ok", "thanks"
}


def _load_detected_stopwords() -> set:
    """Load stop words from cache if available, else use fallback."""
    import json
    cache_file = CACHE_DIR / "detected_stopwords.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            detected = set(data.get('detected_stopwords', []))
            if detected:
                print(f"   [config] Loaded {len(detected)} stop words from cache")
                return detected
        except Exception as e:
            print(f"   [config] Warning: Could not load stop words cache: {e}")
    
    # Fallback
    return STOP_WORDS_FALLBACK


# Load stop words (from cache or fallback)
STOP_WORDS = _load_detected_stopwords()

