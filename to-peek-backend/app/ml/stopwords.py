"""
Automatic stopwords detection via entropy analysis.

Detects corpus-specific stopwords by finding words that appear
uniformly across all clusters (high entropy = no discriminative power).
"""
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

from .config import STOP_WORDS as STATIC_STOP_WORDS


# Parameters for entropy detection
N_CLUSTERS = 10
ENTROPY_THRESHOLD = 0.85  # Ratio of max entropy (1.0 = perfectly uniform)
MAX_RATIO_THRESHOLD = 2.0  # If max(cluster) / mean(cluster) > this, word is "saved"
MIN_WORD_FREQ = 20
MAX_SAMPLE_FOR_DETECTION = 10000  # Limit for performance


def detect_stopwords(
    comments: list[str],
    embeddings: np.ndarray,
    verbose: bool = True,
) -> set[str]:
    """
    Detect corpus-specific stopwords using entropy across clusters.
    
    Words that appear uniformly across all clusters have no discriminative
    power and should be treated as stopwords for topic modeling.
    
    Args:
        comments: List of comment texts
        embeddings: Pre-computed embeddings for comments
        verbose: Print progress
        
    Returns:
        Set of detected stopwords (combined with static list)
    """
    if verbose:
        print("\n🔍 Detecting corpus-specific stopwords...")
    
    # Sample if too large
    if len(comments) > MAX_SAMPLE_FOR_DETECTION:
        indices = np.random.choice(len(comments), MAX_SAMPLE_FOR_DETECTION, replace=False)
        sample_comments = [comments[i] for i in indices]
        sample_embeddings = embeddings[indices]
        if verbose:
            print(f"   Sampled {MAX_SAMPLE_FOR_DETECTION:,} comments for detection")
    else:
        sample_comments = comments
        sample_embeddings = embeddings
    
    # Quick clustering with K-means
    if verbose:
        print(f"   Clustering into {N_CLUSTERS} groups...")
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sample_embeddings)
    
    # Build vocabulary
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b',
        min_df=5,
    )
    
    try:
        doc_term_matrix = vectorizer.fit_transform(sample_comments)
    except ValueError:
        # Not enough documents or vocabulary
        if verbose:
            print("   Not enough vocabulary for detection, using static list only")
        return set(STATIC_STOP_WORDS)
    
    vocabulary = vectorizer.get_feature_names_out()
    if verbose:
        print(f"   Vocabulary size: {len(vocabulary)} words")
    
    # Count word frequency per cluster
    word_cluster_freq = defaultdict(lambda: np.zeros(N_CLUSTERS))
    
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        cluster_docs = doc_term_matrix[cluster_mask]
        word_counts = np.asarray(cluster_docs.sum(axis=0)).flatten()
        
        for word_idx, count in enumerate(word_counts):
            if count > 0:
                word_cluster_freq[vocabulary[word_idx]][cluster_id] = count
    
    # Compute entropy for each word
    max_entropy = np.log(N_CLUSTERS)
    detected = []
    
    for word, cluster_counts in word_cluster_freq.items():
        total_freq = cluster_counts.sum()
        if total_freq < MIN_WORD_FREQ:
            continue
        
        # Normalize to probability distribution
        probs = cluster_counts / total_freq
        probs_nonzero = probs[probs > 0]
        
        word_entropy = entropy(probs_nonzero)
        entropy_ratio = word_entropy / max_entropy  # 0 to 1
        
        # Max ratio: detect asymmetry (peak in one cluster)
        mean_count = cluster_counts.mean()
        max_count = cluster_counts.max()
        max_ratio = max_count / mean_count if mean_count > 0 else 0
        
        # High entropy AND low max_ratio = uniform distribution = stopword
        if entropy_ratio >= ENTROPY_THRESHOLD and max_ratio <= MAX_RATIO_THRESHOLD:
            detected.append(word)
    
    # Combine with static stopwords
    all_stopwords = set(STATIC_STOP_WORDS) | set(detected)
    
    if verbose:
        print(f"   Detected {len(detected)} corpus-specific stopwords")
        if detected:
            preview = ", ".join(sorted(detected)[:15])
            if len(detected) > 15:
                preview += f", ... (+{len(detected) - 15} more)"
            print(f"   Examples: {preview}")
        print(f"   Total stopwords: {len(all_stopwords)} (static: {len(STATIC_STOP_WORDS)}, detected: {len(detected)})")
    
    return all_stopwords

