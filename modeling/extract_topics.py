#!/usr/bin/env python3
"""
Extract topics from YouTube comments using BERTopic.
Takes a random sample of comments for quick experimentation.
"""

import os
# Disable tokenizer parallelism warning (must be before imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import json
import argparse
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# Import from our lib
from lib import (
    # Config
    DATASETS_DIR, STOP_WORDS, EMBEDDING_MODEL_NAME,
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST, UMAP_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_CLUSTER_EPSILON, HDBSCAN_METRIC,
    MIN_TOPIC_SIZE, NR_TOPICS, VECTORIZER_MIN_DF, VECTORIZER_NGRAM_RANGE,
    SEMANTIC_TOP_N_WORDS, SEMANTIC_VOCAB_MIN_DF, SEMANTIC_NGRAM_RANGE,
    SEMANTIC_CANDIDATES, MMR_LAMBDA,
    # Functions
    load_comments,
    get_comments_embeddings_cached,
    get_vocab_embeddings_cached,
    mmr_selection_fast,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract topics from comments")
    parser.add_argument("--size", type=int, default=1000,
                        help="Number of comments to sample (default: 1000)")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    sample_size = args.size
    
    print("=" * 60)
    print("Topic Extraction with BERTopic")
    print("=" * 60)
    
    # 1. Load comments
    print("\n1. Loading comments...")
    all_comments = load_comments()
    if not all_comments:
        return
    print(f"   Total comments available: {len(all_comments):,}")
    
    # 2. Sample
    print(f"\n2. Sampling {sample_size} random comments...")
    random.seed(42)
    sample = random.sample(all_comments, min(sample_size, len(all_comments)))
    sample = [c for c in sample if len(c) > 20]  # Filter short comments
    print(f"   Sample size after filtering: {len(sample)}")
    
    # 3. Create models
    print("\n3. Creating BERTopic model...")
    print(f"   UMAP: n_neighbors={UMAP_N_NEIGHBORS}, n_components={UMAP_N_COMPONENTS}, min_dist={UMAP_MIN_DIST}")
    print(f"   HDBSCAN: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}")
    
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_EPSILON, metric=HDBSCAN_METRIC,
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(
        stop_words=list(STOP_WORDS), min_df=VECTORIZER_MIN_DF,
        ngram_range=VECTORIZER_NGRAM_RANGE
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    topic_model = BERTopic(
        language="multilingual", umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model, min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS, verbose=True
    )
    
    # 4. Get embeddings (cached)
    print("\n4. Getting comment embeddings (with cache)...")
    embeddings = get_comments_embeddings_cached(sample, embedding_model)
    
    # 5. Fit model
    print("\n5. Fitting model to comments...")
    topics, _ = topic_model.fit_transform(sample, embeddings=embeddings)
    
    # 6. Compute cluster metrics
    print("\n6. Computing cluster quality metrics...")
    embeddings_reduced = umap_model.transform(embeddings)
    cluster_metrics = compute_cluster_metrics(
        topics, embeddings, embeddings_reduced, sample,
        embedding_model, STOP_WORDS
    )
    
    # 7. Display results
    display_results(topic_model, cluster_metrics, sample, topics)
    
    # 8. Save results
    save_results(topic_model, cluster_metrics, sample, topics)


def compute_cluster_metrics(topics, embeddings, embeddings_reduced, sample, embedding_model, stop_words):
    """Compute silhouette, variance, and centroid+MMR words for each cluster."""
    
    # Build vocabulary
    print("\n7. Building vocabulary for semantic word extraction...")
    vocab_vectorizer = CountVectorizer(
        stop_words=list(stop_words), min_df=SEMANTIC_VOCAB_MIN_DF,
        ngram_range=SEMANTIC_NGRAM_RANGE,
        token_pattern=r'\b[a-zA-ZÀ-ÿ]{3,}\b'
    )
    vocab_vectorizer.fit(sample)
    vocabulary = list(vocab_vectorizer.vocabulary_.keys())
    print(f"   Vocabulary size: {len(vocabulary)} words")
    
    vocab_embeddings = get_vocab_embeddings_cached(vocabulary, embedding_model)
    print(f"   Vocabulary embeddings shape: {vocab_embeddings.shape}")
    
    # Compute metrics per cluster
    topics_array = np.array(topics)
    non_outlier_mask = topics_array != -1
    cluster_metrics = {}
    
    if non_outlier_mask.sum() > 1:
        silhouette_scores = silhouette_samples(
            embeddings_reduced[non_outlier_mask], 
            topics_array[non_outlier_mask]
        )
        full_silhouette = np.full(len(topics), np.nan)
        full_silhouette[non_outlier_mask] = silhouette_scores
        
        for cluster_id in set(topics):
            if cluster_id == -1:
                continue
            mask = topics_array == cluster_id
            cluster_points = embeddings_reduced[mask]
            
            # Silhouette
            cluster_silhouette = full_silhouette[mask]
            mean_silhouette = float(np.nanmean(cluster_silhouette))
            
            # Variance
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            variance = float(distances.std())
            max_dist = float(distances.max())
            
            # Centroid+MMR words
            cluster_doc_embeddings = embeddings[mask]
            topic_centroid = cluster_doc_embeddings.mean(axis=0).reshape(1, -1)
            similarities = cosine_similarity(topic_centroid, vocab_embeddings)[0]
            
            top_candidate_indices = np.argsort(similarities)[-SEMANTIC_CANDIDATES:][::-1]
            candidate_embeddings = vocab_embeddings[top_candidate_indices]
            
            selected_indices = mmr_selection_fast(
                similarities, candidate_embeddings, top_candidate_indices,
                top_n=SEMANTIC_TOP_N_WORDS, lambda_param=MMR_LAMBDA
            )
            centroid_mmr_words = [(vocabulary[i], float(similarities[i])) for i in selected_indices]
            
            # Centroid+MMR comments (representative + diverse)
            cluster_indices = np.where(mask)[0]
            cluster_comments = [sample[i] for i in cluster_indices]
            doc_similarities = cosine_similarity(topic_centroid, cluster_doc_embeddings)[0]
            
            # Select top 20 candidates, then MMR for diversity
            n_candidates = min(20, len(cluster_comments))
            top_doc_indices = np.argsort(doc_similarities)[-n_candidates:][::-1]
            candidate_doc_embeddings = cluster_doc_embeddings[top_doc_indices]
            
            selected_doc_indices = mmr_selection_fast(
                doc_similarities, candidate_doc_embeddings, top_doc_indices,
                top_n=5, lambda_param=MMR_LAMBDA
            )
            centroid_mmr_comments = [
                (cluster_comments[i], float(doc_similarities[i])) 
                for i in selected_doc_indices
            ]
            
            cluster_metrics[cluster_id] = {
                'silhouette': round(mean_silhouette, 4),
                'variance': round(variance, 4),
                'max_distance': round(max_dist, 4),
                'centroid_mmr_words': centroid_mmr_words,
                'centroid_mmr_comments': centroid_mmr_comments
            }
    
    return cluster_metrics


def display_results(topic_model, cluster_metrics, sample, topics):
    """Print results to console."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics (excluding outliers)")
    print("\n--- Topic Overview ---")
    print(topic_info.to_string())
    
    print("\n--- Topic Quality & Top Words (Centroid+MMR) ---")
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        metrics = cluster_metrics.get(topic_id, {})
        sil = metrics.get('silhouette', 0)
        var = metrics.get('variance', 0)
        words = metrics.get('centroid_mmr_words', [])
        flag = " ⚠️ FOURRE-TOUT?" if sil < 0.1 else ""
        
        print(f"\nTopic {topic_id} [sil={sil:.3f}, var={var:.3f}]{flag}")
        if words:
            display = [f"{w}({s:.2f})" for w, s in words[:8]]
            print(f"  {', '.join(display)}")
    
    print("\n--- Example Comments per Topic ---")
    for topic_id in topic_info['Topic'].tolist()[:6]:
        if topic_id == -1:
            continue
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        print(f"\n[Topic {topic_id}]")
        for comment in topic_comments[:3]:
            preview = comment[:100] + "..." if len(comment) > 100 else comment
            print(f"  • {preview}")


def save_results(topic_model, cluster_metrics, sample, topics):
    """Save results to JSON and model files."""
    topic_info = topic_model.get_topic_info()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATASETS_DIR / f"topics_result_{timestamp}.json"
    
    result = {
        'generated_at': datetime.now().isoformat(),
        'sample_size': len(sample),
        'num_topics': len(topic_info) - 1,
        'params': {
            'umap': {
                'n_neighbors': UMAP_N_NEIGHBORS, 'n_components': UMAP_N_COMPONENTS,
                'min_dist': UMAP_MIN_DIST, 'metric': UMAP_METRIC
            },
            'hdbscan': {
                'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
                'min_samples': HDBSCAN_MIN_SAMPLES,
                'cluster_epsilon': HDBSCAN_CLUSTER_EPSILON, 'metric': HDBSCAN_METRIC
            },
            'bertopic': {'min_topic_size': MIN_TOPIC_SIZE, 'nr_topics': NR_TOPICS}
        },
        'topics': []
    }
    
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        topic_comments = [sample[i] for i, t in enumerate(topics) if t == topic_id]
        metrics = cluster_metrics.get(topic_id, {})
        words = metrics.get('centroid_mmr_words', [])
        mmr_comments = metrics.get('centroid_mmr_comments', [])
        result['topics'].append({
            'id': topic_id,
            'count': len(topic_comments),
            'silhouette': metrics.get('silhouette'),
            'variance': metrics.get('variance'),
            'max_distance': metrics.get('max_distance'),
            'top_words_centroid_mmr': [w for w, _ in words],
            'top_words_centroid_mmr_detail': [{'word': w, 'similarity': round(s, 4)} for w, s in words],
            'example_comments_original': topic_comments[:5],
            'example_comments_centroid_mmr': [c for c, _ in mmr_comments],
            'example_comments_centroid_mmr_detail': [
                {'comment': c, 'similarity': round(s, 4)} for c, s in mmr_comments
            ]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Save model
    model_dir = DATASETS_DIR / "bertopic_model"
    print(f"\n8. Saving BERTopic model to: {model_dir}")
    topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=EMBEDDING_MODEL_NAME)
    
    # Save documents
    docs_file = DATASETS_DIR / "bertopic_docs.json"
    with open(docs_file, 'w', encoding='utf-8') as f:
        json.dump({"documents": sample, "topics": topics}, f, ensure_ascii=False)
    print(f"   Documents saved to: {docs_file}")


if __name__ == "__main__":
    main()
