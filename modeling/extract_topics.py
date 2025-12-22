#!/usr/bin/env python3
"""
Extract topics from YouTube comments using BERTopic.
Takes a random sample of 1000 comments for quick experimentation.
"""

import os
# Disable tokenizer parallelism warning (must be before imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import json
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN


DATASETS_DIR = Path(__file__).parent / "datasets"
CACHE_DIR = Path(__file__).parent / "cache"
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

# Stop words français + anglais (les plus communs)
STOP_WORDS_FR = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "en", "que", "qui",
    "dans", "ce", "il", "ne", "sur", "se", "pas", "plus", "par", "pour", "au", "aux",
    "avec", "son", "sa", "ses", "ou", "mais", "comme", "on", "tout", "nous", "vous",
    "ils", "elle", "elles", "été", "être", "avoir", "fait", "faire", "dit", "dire",
    "cette", "ces", "sont", "ont", "leur", "leurs", "même", "aussi", "bien", "sans",
    "peut", "tous", "après", "ainsi", "donc", "entre", "très", "quand", "depuis",
    "encore", "peu", "ça", "cela", "si", "où", "dont", "ni", "car", "avant", "chez",
    "vers", "comment", "pourquoi", "quoi", "là", "moins", "autre", "autres", "y",
    "deux", "fois", "bon", "bonne", "ici", "là", "alors", "toujours", "jamais",
    "déjà", "trop", "assez", "beaucoup", "plusieurs", "chaque", "quelque", "quelques",
    "moi", "toi", "lui", "eux", "te", "me", "tu", "je", "suis", "es", "soit",
    "fais", "vas", "vais", "allons", "allez", "vont", "ont", "avez", "avons",
    "vraiment", "tellement", "puis", "nan", "oui", "non", "ouais", "ok", "super",
    "merci", "svp", "stp", "lol", "mdr", "ptdr", "haha", "genre", "truc", "chose",
    "juste", "parce", "quand", "comme", "car", "donc", "lorsqu", "afin", "sinon"
}

STOP_WORDS_EN = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "been",
    "being", "has", "had", "did", "does", "doing", "don", "doesn", "didn", "won",
    "wouldn", "shouldn", "couldn", "can", "cannot", "could", "should", "would",
    "might", "must", "shall", "will", "need", "got", "getting", "going", "gonna",
    "yeah", "yes", "yep", "nope", "ok", "okay", "lol", "lmao", "haha", "wow",
    "oh", "ah", "um", "uh", "hey", "hi", "hello", "thanks", "thank", "please",
    "really", "very", "much", "many", "more", "less", "too", "still", "already",
    "here", "where", "why", "when", "while", "though", "although", "however",
    "actually", "basically", "literally", "probably", "maybe", "perhaps", "thing"
}

STOP_WORDS = STOP_WORDS_FR | STOP_WORDS_EN


def parse_sample_size():
    import argparse
    parser = argparse.ArgumentParser(description="Extract topics from comments")
    parser.add_argument("--size", type=int, default=1000,
                        help="Number of comments to sample (default: 1000)")
    args, _ = parser.parse_known_args()
    return args.size

SAMPLE_SIZE = parse_sample_size()


def mmr_selection_fast(centroid_sim: np.ndarray, word_embeddings: np.ndarray, 
                        candidate_indices: np.ndarray, top_n: int, 
                        lambda_param: float = 0.7) -> list:
    """
    Fast Maximal Marginal Relevance selection using vectorized operations.
    
    Args:
        centroid_sim: Similarity scores between words and topic centroid (full vocab)
        word_embeddings: Embeddings of candidate words only (pre-filtered)
        candidate_indices: Original indices of candidates in full vocab
        top_n: Number of words to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of selected original indices
    """
    n_candidates = len(candidate_indices)
    if n_candidates == 0:
        return []
    
    # Pre-compute similarity matrix between all candidates (vectorized)
    candidate_sims = cosine_similarity(word_embeddings)  # N x N matrix
    
    # Relevance scores for candidates
    relevance = centroid_sim[candidate_indices]
    
    # Track selected (local indices within candidates)
    selected_local = [np.argmax(relevance)]
    remaining = set(range(n_candidates)) - set(selected_local)
    
    for _ in range(min(top_n - 1, n_candidates - 1)):
        if not remaining:
            break
        
        remaining_list = list(remaining)
        
        # Max similarity to any already selected word (vectorized lookup)
        max_sim_to_selected = candidate_sims[np.ix_(remaining_list, selected_local)].max(axis=1)
        
        # MMR scores
        mmr_scores = lambda_param * relevance[remaining_list] - (1 - lambda_param) * max_sim_to_selected
        
        # Select best
        best_local = remaining_list[np.argmax(mmr_scores)]
        selected_local.append(best_local)
        remaining.remove(best_local)
    
    # Map back to original indices
    return [candidate_indices[i] for i in selected_local]


def _hash_text(text: str) -> str:
    """Fast hash for cache lookup."""
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_comments_embeddings_cached(comments: list, embedding_model) -> np.ndarray:
    """
    Get embeddings for comments, using cache for already computed ones.
    Uses MD5 hash of text as cache key.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load existing cache
    cache_dict = {}
    if COMMENTS_CACHE_FILE.exists():
        cache_df = pl.read_parquet(COMMENTS_CACHE_FILE)
        for row in cache_df.iter_rows(named=True):
            cache_dict[row['hash']] = np.array(row['embedding'])
        print(f"   Comments cache loaded: {len(cache_dict)} entries")
    
    # Hash all comments
    comment_hashes = [_hash_text(c) for c in comments]
    
    # Find comments not in cache
    new_indices = [i for i, h in enumerate(comment_hashes) if h not in cache_dict]
    new_comments = [comments[i] for i in new_indices]
    new_hashes = [comment_hashes[i] for i in new_indices]
    
    cached_count = len(comments) - len(new_comments)
    print(f"   Comments: {len(comments)} | Cached: {cached_count} | New: {len(new_comments)}")
    
    # Encode only new comments
    if new_comments:
        print(f"   Encoding {len(new_comments)} new comments...")
        new_embeddings = embedding_model.encode(
            new_comments,
            batch_size=32,
            show_progress_bar=len(new_comments) > 100
        )
        
        # Add to cache dict
        for h, emb in zip(new_hashes, new_embeddings):
            cache_dict[h] = emb
        
        # Save updated cache
        cache_data = {
            'hash': list(cache_dict.keys()),
            'embedding': [emb.tolist() for emb in cache_dict.values()]
        }
        cache_df = pl.DataFrame(cache_data)
        cache_df.write_parquet(COMMENTS_CACHE_FILE)
        print(f"   Comments cache updated: {len(cache_dict)} entries total")
    else:
        print("   All comments found in cache!")
    
    # Return embeddings in comments order
    return np.array([cache_dict[h] for h in comment_hashes])


def get_vocab_embeddings_cached(vocabulary: list, embedding_model) -> np.ndarray:
    """
    Get embeddings for vocabulary words, using cache for already computed ones.
    Only encodes new words and updates the cache.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load existing cache
    cache_dict = {}
    if VOCAB_CACHE_FILE.exists():
        cache_df = pl.read_parquet(VOCAB_CACHE_FILE)
        for row in cache_df.iter_rows(named=True):
            cache_dict[row['word']] = np.array(row['embedding'])
        print(f"   Cache loaded: {len(cache_dict)} words")
    
    # Find words not in cache
    vocab_set = set(vocabulary)
    cached_words = set(cache_dict.keys())
    new_words = list(vocab_set - cached_words)
    
    print(f"   Vocabulary: {len(vocabulary)} | Cached: {len(vocab_set & cached_words)} | New: {len(new_words)}")
    
    # Encode only new words
    if new_words:
        print(f"   Encoding {len(new_words)} new words...")
        new_embeddings = embedding_model.encode(
            new_words,
            batch_size=128,
            show_progress_bar=len(new_words) > 100
        )
        
        # Add to cache dict
        for word, emb in zip(new_words, new_embeddings):
            cache_dict[word] = emb
        
        # Save updated cache
        cache_data = {
            'word': list(cache_dict.keys()),
            'embedding': [emb.tolist() for emb in cache_dict.values()]
        }
        cache_df = pl.DataFrame(cache_data)
        cache_df.write_parquet(VOCAB_CACHE_FILE)
        print(f"   Cache updated: {len(cache_dict)} words total")
    else:
        print("   All words found in cache!")
    
    # Return embeddings in vocabulary order
    return np.array([cache_dict[word] for word in vocabulary])


def load_comments():
    """Load comments from the Parquet dataset."""
    parquet_file = DATASETS_DIR / "comments_full.parquet"
    
    if not parquet_file.exists():
        print(f"Dataset not found: {parquet_file}")
        print("Run create_dataset.py first!")
        return []
    
    df = pl.read_parquet(parquet_file)
    comments = df["text"].to_list()
    
    return comments


def main():
    print("=" * 60)
    print("Topic Extraction with BERTopic")
    print("=" * 60)
    
    # Load comments
    print("\n1. Loading comments...")
    all_comments = load_comments()
    
    if not all_comments:
        return
    
    print(f"   Total comments available: {len(all_comments):,}")
    
    # Random sample
    print(f"\n2. Sampling {SAMPLE_SIZE} random comments...")
    random.seed(42)  # For reproducibility
    sample = random.sample(all_comments, min(SAMPLE_SIZE, len(all_comments)))
    print(f"   Sample size: {len(sample)}")
    
    # Filter very short comments (less useful for topic modeling)
    sample = [c for c in sample if len(c) > 20]
    print(f"   After filtering short comments: {len(sample)}")
    
    # Create BERTopic model with explicit UMAP and HDBSCAN
    print("\n3. Creating BERTopic model...")
    print("   (This may take a minute on first run - downloading model)")
    print(f"   UMAP: n_neighbors={UMAP_N_NEIGHBORS}, n_components={UMAP_N_COMPONENTS}, min_dist={UMAP_MIN_DIST}")
    print(f"   HDBSCAN: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}")
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42
    )
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_EPSILON,
        metric=HDBSCAN_METRIC,
        prediction_data=True
    )
    
    # CountVectorizer with stop words to filter out common words
    vectorizer_model = CountVectorizer(
        stop_words=list(STOP_WORDS),
        min_df=VECTORIZER_MIN_DF,
        ngram_range=VECTORIZER_NGRAM_RANGE
    )
    
    # Create embedding model for cache
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # BERTopic without KeyBERTInspired - we use our own centroid+MMR for top words
    topic_model = BERTopic(
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS,
        verbose=True
    )
    
    # Pre-compute embeddings with cache
    print("\n4. Getting comment embeddings (with cache)...")
    embeddings = get_comments_embeddings_cached(sample, embedding_model)
    
    # Fit the model with pre-computed embeddings
    print("\n5. Fitting model to comments...")
    topics, probs = topic_model.fit_transform(sample, embeddings=embeddings)
    
    # Get reduced embeddings for cluster quality analysis
    print("\n6. Computing cluster quality metrics...")
    embeddings_reduced = umap_model.transform(embeddings)
    
    # Build vocabulary for semantic word extraction
    print("\n7. Building vocabulary for semantic word extraction...")
    vocab_vectorizer = CountVectorizer(
        stop_words=list(STOP_WORDS),
        min_df=SEMANTIC_VOCAB_MIN_DF,
        ngram_range=SEMANTIC_NGRAM_RANGE,  # Unigrams, bigrams, trigrams
        token_pattern=r'\b[a-zA-ZÀ-ÿ]{3,}\b'  # At least 3 chars, letters only
    )
    vocab_vectorizer.fit(sample)
    vocabulary = list(vocab_vectorizer.vocabulary_.keys())
    print(f"   Vocabulary size: {len(vocabulary)} words")
    
    # Get vocabulary embeddings (with cache for speed)
    vocab_embeddings = get_vocab_embeddings_cached(vocabulary, embedding_model)
    print(f"   Vocabulary embeddings shape: {vocab_embeddings.shape}")
    
    # Compute silhouette per cluster and intra-cluster variance
    topics_array = np.array(topics)
    non_outlier_mask = topics_array != -1
    
    cluster_metrics = {}
    if non_outlier_mask.sum() > 1:
        # Silhouette scores (only for non-outliers)
        silhouette_scores = silhouette_samples(
            embeddings_reduced[non_outlier_mask], 
            topics_array[non_outlier_mask]
        )
        
        # Map back to full array
        full_silhouette = np.full(len(topics), np.nan)
        full_silhouette[non_outlier_mask] = silhouette_scores
        
        for cluster_id in set(topics):
            if cluster_id == -1:
                continue
            mask = topics_array == cluster_id
            cluster_points = embeddings_reduced[mask]
            
            # Silhouette for this cluster
            cluster_silhouette = full_silhouette[mask]
            mean_silhouette = float(np.nanmean(cluster_silhouette))
            
            # Variance: distance to centroid
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            variance = float(distances.std())
            max_dist = float(distances.max())
            
            # Semantic words: find words closest to cluster centroid with MMR for diversity
            cluster_doc_embeddings = embeddings[mask]  # Original embeddings, not reduced
            topic_centroid = cluster_doc_embeddings.mean(axis=0).reshape(1, -1)
            similarities = cosine_similarity(topic_centroid, vocab_embeddings)[0]
            
            # Pre-filter top candidates for speed (before MMR)
            top_candidate_indices = np.argsort(similarities)[-SEMANTIC_CANDIDATES:][::-1]
            candidate_embeddings = vocab_embeddings[top_candidate_indices]
            
            # Use fast MMR on pre-filtered candidates
            selected_indices = mmr_selection_fast(
                similarities, candidate_embeddings, top_candidate_indices,
                top_n=SEMANTIC_TOP_N_WORDS, 
                lambda_param=MMR_LAMBDA
            )
            semantic_words = [(vocabulary[i], float(similarities[i])) for i in selected_indices]
            
            cluster_metrics[cluster_id] = {
                'silhouette': round(mean_silhouette, 4),
                'variance': round(variance, 4),
                'max_distance': round(max_dist, 4),
                'centroid_mmr_words': semantic_words
            }
    
    # Get topic info
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics (excluding outliers)")
    
    print("\n--- Topic Overview ---")
    print(topic_info.to_string())
    
    # Show top words for each topic with quality metrics
    print("\n--- Topic Quality & Top Words (Centroid+MMR) ---")
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue  # Skip outlier topic
        metrics = cluster_metrics.get(topic_id, {})
        sil = metrics.get('silhouette', 0)
        var = metrics.get('variance', 0)
        centroid_mmr_words = metrics.get('centroid_mmr_words', [])
        
        # Flag potential fourre-tout clusters
        flag = " ⚠️ FOURRE-TOUT?" if sil < 0.1 else ""
        
        print(f"\nTopic {topic_id} [sil={sil:.3f}, var={var:.3f}]{flag}")
        if centroid_mmr_words:
            words_display = [f"{word}({score:.2f})" for word, score in centroid_mmr_words[:8]]
            print(f"  {', '.join(words_display)}")
    
    # Show example comments for top topics
    print("\n--- Example Comments per Topic ---")
    for topic_id in topic_info['Topic'].tolist()[:6]:  # Top 5 + outliers
        if topic_id == -1:
            continue
        
        # Find comments in this topic
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        
        print(f"\n[Topic {topic_id}]")
        for comment in topic_comments[:3]:  # Show 3 examples
            preview = comment[:100] + "..." if len(comment) > 100 else comment
            print(f"  • {preview}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATASETS_DIR / f"topics_result_{timestamp}.json"
    result = {
        'generated_at': datetime.now().isoformat(),
        'sample_size': len(sample),
        'num_topics': len(topic_info) - 1,
        'params': {
            'umap': {
                'n_neighbors': UMAP_N_NEIGHBORS,
                'n_components': UMAP_N_COMPONENTS,
                'min_dist': UMAP_MIN_DIST,
                'metric': UMAP_METRIC
            },
            'hdbscan': {
                'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
                'min_samples': HDBSCAN_MIN_SAMPLES,
                'cluster_epsilon': HDBSCAN_CLUSTER_EPSILON,
                'metric': HDBSCAN_METRIC
            },
            'bertopic': {
                'min_topic_size': MIN_TOPIC_SIZE,
                'nr_topics': NR_TOPICS
            }
        },
        'topics': []
    }
    
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        metrics = cluster_metrics.get(topic_id, {})
        centroid_mmr_words = metrics.get('centroid_mmr_words', [])
        result['topics'].append({
            'id': topic_id,
            'count': len(topic_comments),
            'silhouette': metrics.get('silhouette', None),
            'variance': metrics.get('variance', None),
            'max_distance': metrics.get('max_distance', None),
            'top_words_centroid_mmr': [w for w, _ in centroid_mmr_words],
            'top_words_centroid_mmr_detail': [{'word': w, 'similarity': round(s, 4)} for w, s in centroid_mmr_words],
            'example_comments': topic_comments[:5]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Save the BERTopic model for visualization
    model_dir = DATASETS_DIR / "bertopic_model"
    print(f"\n8. Saving BERTopic model to: {model_dir}")
    topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Save the documents (sample) for visualization
    docs_file = DATASETS_DIR / "bertopic_docs.json"
    with open(docs_file, 'w', encoding='utf-8') as f:
        json.dump({"documents": sample, "topics": topics}, f, ensure_ascii=False)
    
    print(f"   Documents saved to: {docs_file}")
    print("\n   Run 'python visualize_topics.py' to generate visualizations!")


if __name__ == "__main__":
    main()

