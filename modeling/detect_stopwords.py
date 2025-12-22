#!/usr/bin/env python3
"""
Detect stop words automatically from our corpus.
Two approaches:
1. Load known stop words from NLTK (FR, EN, etc.)
2. Detect corpus-specific stop words via entropy across clusters

Saves results to cache/detected_stopwords.json
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.stats import entropy

# Paths
DATASETS_DIR = Path(__file__).parent / "datasets"
CACHE_DIR = Path(__file__).parent / "cache"
OUTPUT_FILE = CACHE_DIR / "detected_stopwords.json"

# Parameters
SAMPLE_SIZE = 20000
N_CLUSTERS = 10  # For entropy detection
ENTROPY_THRESHOLD = 0.85  # Ratio of max entropy (1.0 = perfectly uniform)
MAX_RATIO_THRESHOLD = 2.0  # If max(cluster) / mean(cluster) > this, word is "saved"
MIN_WORD_FREQ = 20  # Minimum occurrences for entropy analysis


def load_comments():
    """Load all comments from the full dataset."""
    comments_file = DATASETS_DIR / "comments_full.parquet"
    if comments_file.exists():
        import polars as pl
        df = pl.read_parquet(comments_file)
        return df['text'].to_list()
    
    txt_file = DATASETS_DIR / "comments.txt"
    if txt_file.exists():
        with open(txt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    print("No dataset found. Run create_dataset.py first.")
    return []


def load_nltk_stopwords() -> dict:
    """Load stop words from NLTK for multiple languages."""
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        
        # Ensure data is downloaded
        try:
            nltk_stopwords.words('french')
        except LookupError:
            print("   Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        languages = {
            'french': 'fr',
            'english': 'en', 
            'spanish': 'es',
            'german': 'de',
            'portuguese': 'pt',
            'italian': 'it'
        }
        
        result = {}
        all_words = set()
        
        for nltk_lang, code in languages.items():
            try:
                words = set(nltk_stopwords.words(nltk_lang))
                result[code] = list(words)
                all_words |= words
                print(f"   {code.upper()}: {len(words)} stop words")
            except OSError:
                print(f"   {code.upper()}: not available")
        
        return {
            'by_language': result,
            'all': list(all_words)
        }
    
    except ImportError:
        print("   NLTK not installed, skipping...")
        return {'by_language': {}, 'all': []}


def detect_entropy_stopwords(comments: list) -> dict:
    """
    Detect stop words by entropy across clusters.
    Words that appear uniformly across all clusters are likely stop words.
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"\n   Embedding {len(comments)} comments...")
    
    # Check cache for embeddings
    embeddings_cache = CACHE_DIR / "comments_embeddings.parquet"
    if embeddings_cache.exists():
        import polars as pl
        import hashlib
        
        cache_df = pl.read_parquet(embeddings_cache)
        cache_dict = {row['hash']: row['embedding'] for row in cache_df.iter_rows(named=True)}
        
        # Get embeddings from cache
        embeddings = []
        uncached_comments = []
        uncached_indices = []
        
        for i, comment in enumerate(comments):
            h = hashlib.md5(comment.encode()).hexdigest()
            if h in cache_dict:
                embeddings.append(cache_dict[h])
            else:
                uncached_comments.append(comment)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Encode uncached
        if uncached_comments:
            print(f"   Found {len(comments) - len(uncached_comments)} in cache, encoding {len(uncached_comments)} new...")
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            new_embeddings = model.encode(uncached_comments, show_progress_bar=True)
            for i, emb in zip(uncached_indices, new_embeddings):
                embeddings[i] = emb
        else:
            print(f"   All {len(comments)} found in cache!")
        
        embeddings = np.array(embeddings)
    else:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(comments, show_progress_bar=True)
    
    # Quick clustering with K-means
    print(f"\n   Clustering into {N_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Show cluster sizes
    cluster_sizes = Counter(cluster_labels)
    print(f"   Cluster sizes: {dict(sorted(cluster_sizes.items()))}")
    
    # Build vocabulary
    print(f"\n   Building vocabulary...")
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b',
        min_df=5
    )
    doc_term_matrix = vectorizer.fit_transform(comments)
    vocabulary = vectorizer.get_feature_names_out()
    print(f"   Vocabulary size: {len(vocabulary)} words")
    
    # Count word frequency per cluster
    print(f"\n   Computing word distribution per cluster...")
    word_cluster_freq = defaultdict(lambda: np.zeros(N_CLUSTERS))
    
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        cluster_docs = doc_term_matrix[cluster_mask]
        word_counts = np.asarray(cluster_docs.sum(axis=0)).flatten()
        
        for word_idx, count in enumerate(word_counts):
            if count > 0:
                word_cluster_freq[vocabulary[word_idx]][cluster_id] = count
    
    # Compute entropy for each word
    print(f"\n   Computing entropy for each word...")
    max_entropy = np.log(N_CLUSTERS)  # Entropy of uniform distribution
    
    word_stats = []
    for word, cluster_counts in word_cluster_freq.items():
        total_freq = cluster_counts.sum()
        if total_freq < MIN_WORD_FREQ:
            continue
        
        # Normalize to probability distribution
        probs = cluster_counts / total_freq
        probs_nonzero = probs[probs > 0]  # Remove zeros for entropy calc
        
        word_entropy = entropy(probs_nonzero)
        entropy_ratio = word_entropy / max_entropy  # 0 to 1
        
        # Max ratio: detect asymmetry (peak in one cluster)
        mean_count = cluster_counts.mean()
        max_count = cluster_counts.max()
        max_ratio = max_count / mean_count if mean_count > 0 else 0
        
        word_stats.append({
            'word': word,
            'entropy': round(float(word_entropy), 4),
            'entropy_ratio': round(float(entropy_ratio), 4),
            'max_ratio': round(float(max_ratio), 2),
            'total_freq': int(total_freq),
            'cluster_distribution': [int(c) for c in cluster_counts]
        })
    
    # Sort by entropy ratio (highest = most uniform = most likely stop word)
    word_stats.sort(key=lambda x: -x['entropy_ratio'])
    
    # Filter: high entropy AND low max_ratio (uniform, no peak)
    detected = []
    saved = []  # Words with high entropy but asymmetric distribution
    for w in word_stats:
        if w['entropy_ratio'] >= ENTROPY_THRESHOLD:
            if w['max_ratio'] <= MAX_RATIO_THRESHOLD:
                detected.append(w)
            else:
                w['saved_reason'] = f"max_ratio={w['max_ratio']:.1f} > {MAX_RATIO_THRESHOLD}"
                saved.append(w)
    
    return {
        'detected': [w['word'] for w in detected],
        'details': detected,
        'saved': saved,  # Words that passed entropy but failed max_ratio
        'all_stats': word_stats[:200],  # Top 200 for inspection
        'params': {
            'n_clusters': N_CLUSTERS,
            'entropy_threshold': ENTROPY_THRESHOLD,
            'max_ratio_threshold': MAX_RATIO_THRESHOLD,
            'min_word_freq': MIN_WORD_FREQ,
            'max_entropy': round(float(max_entropy), 4)
        }
    }


def main():
    print("=" * 60)
    print("Stop Words Detection")
    print("=" * 60)
    
    # 1. Load NLTK stop words
    print("\n1. Loading NLTK stop words...")
    nltk_result = load_nltk_stopwords()
    nltk_total = len(nltk_result['all'])
    print(f"   Total from NLTK: {nltk_total} unique words")
    
    # 2. Load comments
    print("\n2. Loading comments...")
    all_comments = load_comments()
    if not all_comments:
        return
    print(f"   Total: {len(all_comments):,} comments")
    
    # Sample
    if len(all_comments) > SAMPLE_SIZE:
        random.seed(42)
        comments = random.sample(all_comments, SAMPLE_SIZE)
        print(f"   Sampled: {len(comments):,} comments")
    else:
        comments = all_comments
    
    # 3. Detect via entropy
    print("\n3. Detecting corpus-specific stop words via entropy...")
    entropy_result = detect_entropy_stopwords(comments)
    
    # 4. Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n--- NLTK Stop Words ({nltk_total} total) ---")
    for lang, words in nltk_result['by_language'].items():
        preview = ', '.join(sorted(words)[:10])
        print(f"   {lang.upper()}: {preview}...")
    
    print(f"\n--- Entropy-Detected Stop Words ({len(entropy_result['detected'])}) ---")
    print(f"   (entropy_ratio >= {ENTROPY_THRESHOLD} AND max_ratio <= {MAX_RATIO_THRESHOLD})")
    
    for item in entropy_result['details'][:30]:
        dist_str = ''.join(['█' if c > 0 else '·' for c in item['cluster_distribution']])
        print(f"   {item['word']:15} ent={item['entropy_ratio']:.2f}  max_r={item['max_ratio']:.1f}  freq={item['total_freq']:>5}  [{dist_str}]")
    
    if len(entropy_result['details']) > 30:
        print(f"   ... and {len(entropy_result['details']) - 30} more")
    
    # Show saved words (high entropy but asymmetric)
    saved = entropy_result.get('saved', [])
    if saved:
        print(f"\n--- SAVED Words ({len(saved)}) - high entropy but asymmetric ---")
        for item in saved[:20]:
            dist_str = ''.join(['█' if c > 0 else '·' for c in item['cluster_distribution']])
            print(f"   ✓ {item['word']:15} ent={item['entropy_ratio']:.2f}  max_r={item['max_ratio']:.1f}  freq={item['total_freq']:>5}  [{dist_str}]")
        if len(saved) > 20:
            print(f"   ... and {len(saved) - 20} more saved")
    
    print(f"\n--- Near-threshold words (for inspection) ---")
    near_threshold = [w for w in entropy_result['all_stats'] 
                      if 0.7 <= w['entropy_ratio'] < ENTROPY_THRESHOLD][:20]
    for item in near_threshold:
        dist_str = ''.join(['█' if c > 0 else '·' for c in item['cluster_distribution']])
        print(f"   {item['word']:15} entropy={item['entropy_ratio']:.3f}  freq={item['total_freq']:>5}  [{dist_str}]")
    
    # 5. Combine and save
    all_stopwords = set(nltk_result['all']) | set(entropy_result['detected'])
    
    result = {
        'detected_stopwords': sorted(list(all_stopwords)),
        'sources': {
            'nltk': {
                'count': nltk_total,
                'by_language': {k: len(v) for k, v in nltk_result['by_language'].items()}
            },
            'entropy': {
                'count': len(entropy_result['detected']),
                'words': entropy_result['detected'],
                'details': entropy_result['details']
            }
        },
        'params': entropy_result['params']
    }
    
    CACHE_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"SAVED: {OUTPUT_FILE}")
    print(f"   NLTK: {nltk_total} words")
    print(f"   Entropy-detected: {len(entropy_result['detected'])} words")
    print(f"   Total unique: {len(all_stopwords)} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
