#!/usr/bin/env python3
"""
Extract topics from YouTube comments using BERTopic.
Takes a random sample of 1000 comments for quick experimentation.
"""

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
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN


DATASETS_DIR = Path(__file__).parent / "datasets"

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
    
    # KeyBERT-Inspired gives better keyword extraction for topics
    representation_model = KeyBERTInspired()
    
    topic_model = BERTopic(
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS,
        verbose=True
    )
    
    # Fit the model
    print("\n4. Fitting model to comments...")
    topics, probs = topic_model.fit_transform(sample)
    
    # Get embeddings for cluster quality analysis
    print("\n5. Computing cluster quality metrics...")
    embeddings = topic_model._extract_embeddings(sample, method="document")
    embeddings_reduced = umap_model.transform(embeddings)
    
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
            
            cluster_metrics[cluster_id] = {
                'silhouette': round(mean_silhouette, 4),
                'variance': round(variance, 4),
                'max_distance': round(max_dist, 4)
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
    print("\n--- Topic Quality & Top Words ---")
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue  # Skip outlier topic
        topic_words = topic_model.get_topic(topic_id)
        metrics = cluster_metrics.get(topic_id, {})
        sil = metrics.get('silhouette', 0)
        var = metrics.get('variance', 0)
        
        # Flag potential fourre-tout clusters
        flag = " ⚠️ FOURRE-TOUT?" if sil < 0.1 else ""
        
        if topic_words:
            words = [word for word, _ in topic_words[:8]]
            print(f"\nTopic {topic_id} [sil={sil:.3f}, var={var:.3f}]{flag}: {', '.join(words)}")
    
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
        topic_words = topic_model.get_topic(topic_id)
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        metrics = cluster_metrics.get(topic_id, {})
        result['topics'].append({
            'id': topic_id,
            'count': len(topic_comments),
            'silhouette': metrics.get('silhouette', None),
            'variance': metrics.get('variance', None),
            'max_distance': metrics.get('max_distance', None),
            'top_words': [word for word, _ in topic_words[:10]] if topic_words else [],
            'example_comments': topic_comments[:5]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Save the BERTopic model for visualization
    model_dir = DATASETS_DIR / "bertopic_model"
    print(f"\n6. Saving BERTopic model to: {model_dir}")
    topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Save the documents (sample) for visualization
    docs_file = DATASETS_DIR / "bertopic_docs.json"
    with open(docs_file, 'w', encoding='utf-8') as f:
        json.dump({"documents": sample, "topics": topics}, f, ensure_ascii=False)
    
    print(f"   Documents saved to: {docs_file}")
    print("\n   Run 'python visualize_topics.py' to generate visualizations!")


if __name__ == "__main__":
    main()

