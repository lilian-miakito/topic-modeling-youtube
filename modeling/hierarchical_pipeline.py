#!/usr/bin/env python3
"""
Hierarchical Topic Extraction Pipeline

Elegant pipeline that:
1. Extracts initial topics
2. Names them
3. Identifies low-silhouette (fourre-tout) clusters
4. Re-clusters those into sub-topics
5. Names sub-topics with parent context
6. Saves complete hierarchical structure
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SILHOUETTE_THRESHOLD = 0.15     # Below this → split into sub-topics
MAX_DEPTH = 1                   # Max nesting levels (1 = parent + children)
MIN_CLUSTER_SIZE_FOR_SPLIT = 30 # Don't split tiny clusters

# Extraction params
SAMPLE_SIZE = 10000
MIN_TOPIC_SIZE = 15

# Naming
NUM_PARALLEL_WORKERS = 10
NUM_TOP_WORDS = 15
NUM_COMMENTS = 6

# =============================================================================
# IMPORTS (after config to allow clean error messages)
# =============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lib import DATASETS_DIR, load_comments
from lib.config import UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, UMAP_MIN_DIST, UMAP_METRIC
from lib.config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC
from lib.cache import get_comments_embeddings_cached
from lib.topic_namer import configure_dspy, create_optimized_namer, name_subtopic

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_samples


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class HierarchicalTopicPipeline:
    """
    Orchestrates hierarchical topic extraction and naming.
    """
    
    def __init__(self, model_name: str = "openai/gpt-5-mini"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.namer = None
        self.subtopic_namer = None
        
    def run(self):
        """Execute the full pipeline."""
        print("=" * 70)
        print("🌳 HIERARCHICAL TOPIC EXTRACTION PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        sample, embeddings = self._load_data()
        
        # Step 2: Initial extraction
        topics, topic_model = self._extract_topics(sample, embeddings)
        
        # Step 3: Compute metrics
        topics = self._compute_metrics(topics, sample, embeddings, topic_model)
        
        # Step 4: Configure naming
        self._configure_naming()
        
        # Step 5: Name top-level topics
        topics = self._name_topics(topics, parent_name=None)
        
        # Step 6: Identify fourre-tout clusters and split
        topics = self._split_low_silhouette_topics(topics, sample, embeddings)
        
        # Step 7: Save results
        self._save_results(topics, sample)
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        
        return topics
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    
    def _load_data(self):
        """Load comments and embeddings."""
        print("\n📚 Step 1: Loading data...")
        
        df = load_comments()
        print(f"   Total comments: {len(df):,}")
        
        # Sample
        if len(df) > SAMPLE_SIZE:
            sample = df.sample(n=SAMPLE_SIZE, random_state=42)['text'].tolist()
        else:
            sample = df['text'].tolist()
        print(f"   Sample size: {len(sample):,}")
        
        # Embeddings (cached)
        embeddings = get_comments_embeddings_cached(sample, self.embedding_model)
        print(f"   Embeddings shape: {embeddings.shape}")
        
        return sample, embeddings
    
    # -------------------------------------------------------------------------
    # Step 2: Extract Topics
    # -------------------------------------------------------------------------
    
    def _extract_topics(self, sample, embeddings):
        """Run BERTopic extraction."""
        print("\n🔬 Step 2: Extracting topics...")
        
        umap_model = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric=HDBSCAN_METRIC,
            prediction_data=True
        )
        
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=MIN_TOPIC_SIZE,
            calculate_probabilities=False,
            verbose=False
        )
        
        topic_labels, _ = topic_model.fit_transform(sample, embeddings)
        
        # Build topic list
        topics = []
        unique_labels = set(topic_labels) - {-1}
        
        for topic_id in sorted(unique_labels):
            mask = np.array(topic_labels) == topic_id
            topic_comments = [sample[i] for i in range(len(sample)) if mask[i]]
            topic_embeddings = embeddings[mask]
            
            topics.append({
                'id': topic_id,
                'depth': 0,
                'parent_id': None,
                'count': len(topic_comments),
                'comments': topic_comments,
                'embeddings': topic_embeddings,
                'indices': np.where(mask)[0].tolist()
            })
        
        print(f"   Found {len(topics)} topics (excluding noise)")
        return topics, topic_model
    
    # -------------------------------------------------------------------------
    # Step 3: Compute Metrics
    # -------------------------------------------------------------------------
    
    def _compute_metrics(self, topics, _sample, embeddings, topic_model):
        """Compute silhouette and other metrics for each topic."""
        print("\n📊 Step 3: Computing metrics...")
        
        # Get all labels for silhouette
        labels = topic_model.topics_
        
        # Filter out noise for silhouette
        valid_mask = np.array(labels) != -1
        if valid_mask.sum() > 1:
            valid_embeddings = embeddings[valid_mask]
            valid_labels = np.array(labels)[valid_mask]
            
            silhouette_per_sample = silhouette_samples(valid_embeddings, valid_labels)
            
            # Map back to topics
            sample_idx = 0
            for topic in topics:
                topic_size = topic['count']
                topic_silhouettes = silhouette_per_sample[sample_idx:sample_idx + topic_size]
                
                topic['silhouette'] = float(np.mean(topic_silhouettes))
                topic['variance'] = float(np.var(topic['embeddings'], axis=0).mean())
                
                # Centroid and max distance
                centroid = np.mean(topic['embeddings'], axis=0)
                distances = np.linalg.norm(topic['embeddings'] - centroid, axis=1)
                topic['max_distance'] = float(np.max(distances))
                
                sample_idx += topic_size
        
        # Get top words from BERTopic
        for topic in topics:
            bt_words = topic_model.get_topic(topic['id'])
            if bt_words:
                topic['top_words_ctfidf'] = [w for w, _ in bt_words[:15]]
            else:
                topic['top_words_ctfidf'] = []
        
        fourre_tout = sum(1 for t in topics if t.get('silhouette', 1) < SILHOUETTE_THRESHOLD)
        print(f"   Topics with silhouette < {SILHOUETTE_THRESHOLD}: {fourre_tout}")
        
        return topics
    
    # -------------------------------------------------------------------------
    # Step 4: Configure Naming
    # -------------------------------------------------------------------------
    
    def _configure_naming(self):
        """Configure DSPy for naming."""
        print("\n🏷️  Step 4: Configuring naming...")
        
        configure_dspy(self.model_name)
        self.namer = create_optimized_namer(max_examples=10)
        
        print(f"   Model: {self.model_name}")
    
    # -------------------------------------------------------------------------
    # Step 5: Name Topics
    # -------------------------------------------------------------------------
    
    def _name_topics(self, topics, parent_name=None):
        """Name topics in parallel."""
        level = "sub-topics" if parent_name else "top-level topics"
        print(f"\n🏷️  Step 5: Naming {len(topics)} {level}...")
        
        def name_single(topic):
            keywords = topic.get('top_words_ctfidf', [])[:NUM_TOP_WORDS]
            comments = topic.get('comments', [])[:NUM_COMMENTS]
            
            if not keywords:
                return "Unknown Topic"
            
            try:
                if parent_name:
                    # Sub-topic naming with parent context
                    return name_subtopic(parent_name, keywords, comments)
                else:
                    return self.namer(keywords=keywords, comments=comments)
            except Exception as e:  # noqa: BLE001
                print(f"   Error naming topic {topic['id']}: {e}")
                return f"Topic {topic['id']}"
        
        # Parallel naming
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(name_single, t): i for i, t in enumerate(topics)}
            
            for future in as_completed(futures):
                idx = futures[future]
                topics[idx]['generated_name'] = future.result()
                
                # Progress
                topic = topics[idx]
                print(f"   [{idx+1}/{len(topics)}] {topic['id']}: {topic['generated_name']}")
        
        return topics
    
    # -------------------------------------------------------------------------
    # Step 6: Split Low-Silhouette Topics
    # -------------------------------------------------------------------------
    
    def _split_low_silhouette_topics(self, topics, _sample, _embeddings):
        """Identify and split fourre-tout clusters."""
        print(f"\n🔀 Step 6: Splitting low-silhouette topics (threshold={SILHOUETTE_THRESHOLD})...")
        
        topics_to_split = [
            t for t in topics 
            if t.get('silhouette', 1) < SILHOUETTE_THRESHOLD 
            and t['count'] >= MIN_CLUSTER_SIZE_FOR_SPLIT
            and t['depth'] < MAX_DEPTH
        ]
        
        if not topics_to_split:
            print("   No topics need splitting!")
            return topics
        
        print(f"   Found {len(topics_to_split)} topics to split")
        
        for parent_topic in topics_to_split:
            parent_id = parent_topic['id']
            parent_name = parent_topic.get('generated_name', f"Topic {parent_id}")
            
            print(f"\n   Splitting topic {parent_id} ({parent_name}, sil={parent_topic['silhouette']:.3f})...")
            
            # Get comments and embeddings for this cluster
            cluster_comments = parent_topic['comments']
            cluster_embeddings = parent_topic['embeddings']
            
            if len(cluster_comments) < MIN_CLUSTER_SIZE_FOR_SPLIT:
                print(f"      Too few comments ({len(cluster_comments)}), skipping")
                continue
            
            # Re-cluster with smaller parameters
            sub_umap = UMAP(
                n_neighbors=min(15, len(cluster_comments) - 1),
                n_components=min(5, len(cluster_comments) - 2),
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            sub_hdbscan = HDBSCAN(
                min_cluster_size=max(5, len(cluster_comments) // 10),
                min_samples=3,
                metric='euclidean',
                prediction_data=True
            )
            
            sub_topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=sub_umap,
                hdbscan_model=sub_hdbscan,
                min_topic_size=max(5, len(cluster_comments) // 15),
                calculate_probabilities=False,
                verbose=False
            )
            
            try:
                sub_labels, _ = sub_topic_model.fit_transform(cluster_comments, cluster_embeddings)
            except Exception as e:  # noqa: BLE001
                print(f"      Error during sub-clustering: {e}")
                continue
            
            # Build sub-topics
            sub_topics = []
            unique_sub = set(sub_labels) - {-1}
            
            for sub_id in sorted(unique_sub):
                mask = np.array(sub_labels) == sub_id
                sub_comments = [cluster_comments[i] for i in range(len(cluster_comments)) if mask[i]]
                sub_embeddings = cluster_embeddings[mask]
                
                # Get top words
                bt_words = sub_topic_model.get_topic(sub_id)
                top_words = [w for w, _ in bt_words[:15]] if bt_words else []
                
                # Compute metrics
                if len(sub_comments) > 1:
                    centroid = np.mean(sub_embeddings, axis=0)
                    variance = float(np.var(sub_embeddings, axis=0).mean())
                    distances = np.linalg.norm(sub_embeddings - centroid, axis=1)
                    max_dist = float(np.max(distances))
                else:
                    variance = 0
                    max_dist = 0
                
                sub_topics.append({
                    'id': f"{parent_id}.{sub_id}",
                    'depth': 1,
                    'parent_id': parent_id,
                    'parent_name': parent_name,
                    'count': len(sub_comments),
                    'comments': sub_comments,
                    'embeddings': sub_embeddings,
                    'top_words_ctfidf': top_words,
                    'variance': variance,
                    'max_distance': max_dist
                })
            
            if not sub_topics:
                print("      No sub-topics found")
                continue
            
            print(f"      Found {len(sub_topics)} sub-topics")
            
            # Name sub-topics with parent context
            sub_topics = self._name_topics(sub_topics, parent_name=parent_name)
            
            # Attach to parent
            parent_topic['children'] = sub_topics
            parent_topic['is_hierarchical'] = True
        
        return topics
    
    # -------------------------------------------------------------------------
    # Step 7: Save Results
    # -------------------------------------------------------------------------
    
    def _save_results(self, topics, sample):
        """Save final results to JSON."""
        print("\n💾 Step 7: Saving results...")
        
        # Clean up for JSON serialization
        def clean_topic(t):
            cleaned = {
                'id': t['id'],
                'depth': t.get('depth', 0),
                'parent_id': t.get('parent_id'),
                'parent_name': t.get('parent_name'),
                'generated_name': t.get('generated_name', f"Topic {t['id']}"),
                'count': t['count'],
                'silhouette': t.get('silhouette'),
                'variance': t.get('variance'),
                'max_distance': t.get('max_distance'),
                'top_words_ctfidf': t.get('top_words_ctfidf', []),
                'example_comments': t.get('comments', [])[:10],
                'is_hierarchical': t.get('is_hierarchical', False)
            }
            
            if 'children' in t:
                cleaned['children'] = [clean_topic(c) for c in t['children']]
            
            return cleaned
        
        result = {
            'generated_at': datetime.now().isoformat(),
            'pipeline': 'hierarchical',
            'config': {
                'silhouette_threshold': SILHOUETTE_THRESHOLD,
                'max_depth': MAX_DEPTH,
                'sample_size': len(sample),
                'model': self.model_name
            },
            'num_topics': len(topics),
            'num_hierarchical': sum(1 for t in topics if t.get('is_hierarchical')),
            'topics': [clean_topic(t) for t in topics]
        }
        
        # Count sub-topics
        total_subtopics = sum(len(t.get('children', [])) for t in topics)
        result['num_subtopics'] = total_subtopics
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = DATASETS_DIR / f"topics_hierarchical_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"   Saved to: {output_file.name}")
        print(f"   Topics: {len(topics)}, Hierarchical: {result['num_hierarchical']}, Sub-topics: {total_subtopics}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    model = os.environ.get("DSPY_MODEL", "openai/gpt-5-mini")
    pipeline = HierarchicalTopicPipeline(model_name=model)
    pipeline.run()


if __name__ == "__main__":
    main()

