#!/usr/bin/env python3
"""
Parameter sweep for BERTopic clustering.
Tests different UMAP/HDBSCAN configurations to find optimal number of topics.
"""

import random
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from lib import DATASETS_DIR, load_comments as lib_load_comments

# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

SAMPLE_SIZE = 2000  # Keep small for fast sweeps

# Parameter grids to sweep
UMAP_N_NEIGHBORS_GRID = [10, 15, 30]
UMAP_N_COMPONENTS_GRID = [5, 10]
UMAP_MIN_DIST_GRID = [0.0, 0.1]

HDBSCAN_MIN_CLUSTER_SIZE_GRID = [10, 20, 50]
HDBSCAN_MIN_SAMPLES_GRID = [5, 10]

# =============================================================================


def load_comments():
    """Load comments from the dataset."""
    return lib_load_comments()


def evaluate_clustering(embeddings, labels):
    """Evaluate clustering quality."""
    # Filter out outliers (-1) for silhouette score
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = sum(labels == -1)
    outlier_ratio = n_outliers / len(labels)
    
    # Silhouette score (only if we have at least 2 clusters and some non-outliers)
    if n_clusters >= 2 and sum(mask) > n_clusters:
        try:
            sil_score = silhouette_score(embeddings[mask], labels[mask])
        except:
            sil_score = -1
    else:
        sil_score = -1
    
    return {
        'n_clusters': int(n_clusters),
        'n_outliers': int(n_outliers),
        'outlier_ratio': round(outlier_ratio, 3),
        'silhouette_score': round(sil_score, 4) if sil_score != -1 else None
    }


def run_sweep(comments, embeddings):
    """Run parameter sweep."""
    results = []
    
    # Generate all parameter combinations
    umap_params = list(product(
        UMAP_N_NEIGHBORS_GRID,
        UMAP_N_COMPONENTS_GRID,
        UMAP_MIN_DIST_GRID
    ))
    
    hdbscan_params = list(product(
        HDBSCAN_MIN_CLUSTER_SIZE_GRID,
        HDBSCAN_MIN_SAMPLES_GRID
    ))
    
    total_combinations = len(umap_params) * len(hdbscan_params)
    print(f"Testing {total_combinations} parameter combinations...")
    print("-" * 80)
    
    run_idx = 0
    for umap_nn, umap_nc, umap_md in umap_params:
        # Run UMAP once per UMAP config
        print(f"\nUMAP: n_neighbors={umap_nn}, n_components={umap_nc}, min_dist={umap_md}")
        
        umap_model = UMAP(
            n_neighbors=umap_nn,
            n_components=umap_nc,
            min_dist=umap_md,
            metric='cosine',
            random_state=42
        )
        
        try:
            reduced = umap_model.fit_transform(embeddings)
        except Exception as e:
            print(f"  UMAP failed: {e}")
            continue
        
        for hdb_mcs, hdb_ms in hdbscan_params:
            run_idx += 1
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=hdb_mcs,
                min_samples=hdb_ms,
                metric='euclidean'
            )
            
            try:
                labels = hdbscan_model.fit_predict(reduced)
            except Exception as e:
                print(f"  HDBSCAN failed: {e}")
                continue
            
            metrics = evaluate_clustering(reduced, labels)
            
            result = {
                'run': run_idx,
                'umap_n_neighbors': umap_nn,
                'umap_n_components': umap_nc,
                'umap_min_dist': umap_md,
                'hdbscan_min_cluster_size': hdb_mcs,
                'hdbscan_min_samples': hdb_ms,
                **metrics
            }
            results.append(result)
            
            sil_str = f"{metrics['silhouette_score']:.3f}" if metrics['silhouette_score'] else "N/A"
            print(f"  [{run_idx}/{total_combinations}] HDBSCAN(mcs={hdb_mcs}, ms={hdb_ms}): "
                  f"{metrics['n_clusters']} topics, {metrics['outlier_ratio']*100:.1f}% outliers, "
                  f"silhouette={sil_str}")
    
    return results


def main():
    print("=" * 80)
    print("BERTopic Parameter Sweep")
    print("=" * 80)
    
    # Load comments
    print("\n1. Loading comments...")
    all_comments = load_comments()
    if not all_comments:
        return
    
    print(f"   Total available: {len(all_comments):,}")
    
    # Sample
    random.seed(42)
    sample = random.sample(all_comments, min(SAMPLE_SIZE, len(all_comments)))
    sample = [c for c in sample if len(c) > 20]
    print(f"   Sample size: {len(sample)}")
    
    # Create embeddings
    print("\n2. Creating embeddings...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(sample, show_progress_bar=True)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Run sweep
    print("\n3. Running parameter sweep...")
    results = run_sweep(sample, embeddings)
    
    # Sort by silhouette score (best first)
    results_sorted = sorted(
        [r for r in results if r['silhouette_score'] is not None],
        key=lambda x: x['silhouette_score'],
        reverse=True
    )
    
    # Print top results
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by silhouette score)")
    print("=" * 80)
    
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"\n{i}. Silhouette: {r['silhouette_score']:.4f}")
        print(f"   Topics: {r['n_clusters']}, Outliers: {r['outlier_ratio']*100:.1f}%")
        print(f"   UMAP: n_neighbors={r['umap_n_neighbors']}, n_components={r['umap_n_components']}, min_dist={r['umap_min_dist']}")
        print(f"   HDBSCAN: min_cluster_size={r['hdbscan_min_cluster_size']}, min_samples={r['hdbscan_min_samples']}")
    
    # Also show configs with different topic counts
    print("\n" + "=" * 80)
    print("BEST CONFIG PER TOPIC COUNT")
    print("=" * 80)
    
    by_topics = {}
    for r in results:
        n = r['n_clusters']
        if n not in by_topics or (r['silhouette_score'] or -1) > (by_topics[n]['silhouette_score'] or -1):
            by_topics[n] = r
    
    for n in sorted(by_topics.keys()):
        r = by_topics[n]
        sil = f"{r['silhouette_score']:.4f}" if r['silhouette_score'] else "N/A"
        print(f"  {n:3d} topics: silhouette={sil}, outliers={r['outlier_ratio']*100:.1f}% | "
              f"UMAP({r['umap_n_neighbors']},{r['umap_n_components']},{r['umap_min_dist']}) "
              f"HDBSCAN({r['hdbscan_min_cluster_size']},{r['hdbscan_min_samples']})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = DATASETS_DIR / f"sweep_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'sample_size': len(sample),
            'results': results,
            'best_by_silhouette': results_sorted[:10] if results_sorted else [],
            'best_by_topic_count': by_topics
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

