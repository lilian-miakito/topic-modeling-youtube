"""
Topic extraction service.
Implements the hierarchical topic extraction pipeline.
"""
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

# Disable tokenizer parallelism warning (must be before imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

# Suppress joblib/loky resource tracker warnings (benign cleanup noise)
warnings.filterwarnings("ignore", message="resource_tracker:")
warnings.filterwarnings("ignore", message=".*leaked semaphore.*")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.db.models import Channel, Video, Comment, Extraction
from app.ml import (
    EMBEDDING_MODEL_NAME,
    # PCA pre-reduction (hybrid)
    PCA_N_COMPONENTS,
    # UMAP (clustering)
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    # UMAP (visualization 2D)
    VIZ_UMAP_N_NEIGHBORS,
    VIZ_UMAP_MIN_DIST,
    VIZ_UMAP_METRIC,
    # HDBSCAN (permissive)
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_CLUSTER_EPSILON,
    HDBSCAN_METRIC,
    # BERTopic reduction
    MIN_TOPIC_SIZE,
    TARGET_TOPICS_LEVEL_0,
    OUTLIER_REDUCTION_STRATEGY,
    OUTLIER_REDUCTION_THRESHOLD,
    # Hierarchy (mean-distance-based)
    MEAN_DISTANCE_THRESHOLD,
    MAX_DEPTH,
    # Semantic
    SEMANTIC_TOP_N_WORDS,
    SEMANTIC_CANDIDATES,
    SEMANTIC_VOCAB_MIN_DF,
    SEMANTIC_NGRAM_RANGE,
    MMR_LAMBDA,
    STOP_WORDS,
    # Classes/functions
    EmbeddingsCache,
    mmr_selection_fast,
    detect_stopwords,
    get_adaptive_sub_params,
    get_embedding_model,
    UMAPCache,
)
from app.ml.topic_namer import configure_dspy, get_topic_namer, get_subtopic_namer


# Lazy imports for heavy ML libs
_umap = None
_hdbscan = None
_bertopic = None


def _get_sentence_transformer():
    """Get the SentenceTransformer singleton from warmup module."""
    return get_embedding_model()


def _get_umap():
    global _umap
    if _umap is None:
        from umap import UMAP
        _umap = UMAP
    return _umap


def _get_hdbscan():
    global _hdbscan
    if _hdbscan is None:
        from hdbscan import HDBSCAN
        _hdbscan = HDBSCAN
    return _hdbscan


def _get_bertopic():
    global _bertopic
    if _bertopic is None:
        from bertopic import BERTopic
        _bertopic = BERTopic
    return _bertopic


class ExtractionService:
    """
    Service for hierarchical topic extraction.
    
    Pipeline:
    1. Load comments from DB
    2. Generate embeddings (cached)
    3. Run UMAP + HDBSCAN clustering
    4. Calculate DBCV + persistence scores
    5. Split dispersed clusters (high mean_distance)
    6. Name topics with LLM
    7. Save results
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_model = None
        self.embeddings_cache = None
    
    def _init_ml(self):
        """Initialize ML components lazily."""
        if self.embedding_model is None:
            self.embedding_model = _get_sentence_transformer()
            self.embeddings_cache = EmbeddingsCache(self.db, self.embedding_model, EMBEDDING_MODEL_NAME)
    
    def start_extraction(
        self,
        channel_id: int,
        video_ids: list[int],
        config: dict = None,
    ) -> Extraction:
        """
        Start a new extraction job.
        
        Args:
            channel_id: Channel database ID
            video_ids: List of video IDs to include
            config: Optional extraction configuration
            
        Returns:
            Extraction model instance
        """
        # Create extraction record
        extraction = Extraction(
            channel_id=channel_id,
            video_ids=video_ids,
            config=config or {},
            status="pending",
        )
        self.db.add(extraction)
        self.db.commit()
        self.db.refresh(extraction)
        
        return extraction
    
    def run_extraction(self, extraction_id: int) -> dict:
        """
        Run the full extraction pipeline.
        
        Args:
            extraction_id: Extraction job ID
            
        Returns:
            Result dict with topics
        """
        extraction = self.db.query(Extraction).get(extraction_id)
        if not extraction:
            raise ValueError(f"Extraction {extraction_id} not found")
        
        # Load user config (with defaults from global config)
        user_config = extraction.config or {}
        self.target_topics = user_config.get("num_topics", TARGET_TOPICS_LEVEL_0)
        self.split_threshold = user_config.get("split_threshold", MEAN_DISTANCE_THRESHOLD)
        
        try:
            # Update status
            extraction.status = "running"
            extraction.started_at = datetime.utcnow()
            extraction.current_step = "Loading comments"
            self.db.commit()
            
            print("=" * 70)
            print("🌳 HIERARCHICAL TOPIC EXTRACTION PIPELINE")
            print("=" * 70)
            print(f"   Config: num_topics={self.target_topics}, split_threshold={self.split_threshold}")
            
            # Step 1: Load comments
            print("\n📚 Step 1: Loading comments...")
            comments = self._load_comments(extraction.video_ids)
            if not comments:
                raise ValueError("No comments found for selected videos")
            
            print(f"   Total comments loaded: {len(comments):,}")
            print(f"   Videos: {len(extraction.video_ids)}")
            
            extraction.num_comments = len(comments)
            extraction.progress = 0.1
            extraction.current_step = "Generating embeddings"
            self.db.commit()
            
            # Step 2: Initialize ML and get embeddings
            print("\n🧠 Step 2: Generating embeddings...")
            self._init_ml()
            embeddings = self.embeddings_cache.get_comment_embeddings(comments)
            print(f"   Embeddings shape: {embeddings.shape}")
            
            extraction.progress = 0.25
            extraction.current_step = "Detecting stopwords"
            self.db.commit()
            
            # Step 3: Detect corpus-specific stopwords
            stopwords = detect_stopwords(comments, embeddings, verbose=True)
            
            extraction.progress = 0.35
            extraction.current_step = "Clustering"
            self.db.commit()
            
            # Step 4: Run clustering
            print("\n🔬 Step 4: Running BERTopic clustering...")
            topics, topic_model, outliers_info, umap_embeddings_5d = self._run_clustering(
                comments, embeddings, stopwords, channel_id=extraction.channel_id
            )
            print(f"   Found {len(topics)} initial topics")
            
            extraction.progress = 0.5
            extraction.current_step = "Computing metrics"
            self.db.commit()
            
            # Step 5: Compute metrics (DBCV + persistence)
            print("\n📊 Step 5: Computing DBCV and persistence metrics...")
            topics = self._compute_metrics(topics, embeddings, comments, topic_model)
            
            # Log topic quality
            topics_to_split = sum(1 for t in topics if self._should_split_topic(t))
            print(f"   Topics needing split (mean_distance > {self.split_threshold}): {topics_to_split}")
            for t in topics:
                dist = t.get("mean_distance")
                dist_str = f"dist={dist:.3f}" if dist is not None else "dist=N/A"
                should_split = self._should_split_topic(t)
                marker = "⚠️ " if should_split else "✓ "
                print(f"   {marker}Topic {t['id']}: count={t['count']}, {dist_str}")
            
            extraction.progress = 0.55
            extraction.current_step = "Extracting semantic words"
            self.db.commit()
            
            # Step 6: Extract semantic words (centroid + MMR)
            print("\n🎯 Step 6: Extracting semantic words (centroid + MMR)...")
            topics = self._extract_semantic_words(topics, comments, stopwords)
            
            extraction.progress = 0.65
            extraction.current_step = "Splitting low-quality clusters"
            self.db.commit()
            
            # Step 7: Split dispersed clusters
            print(f"\n🔀 Step 7: Splitting dispersed topics (mean_distance > {self.split_threshold})...")
            topics = self._split_low_quality_topics(topics, embeddings, comments)
            
            extraction.progress = 0.75
            extraction.current_step = "Computing 2D projection"
            self.db.commit()
            
            # Step 8: Compute 2D visualization coordinates (using 5D UMAP embeddings)
            print("\n🗺️  Step 8: Computing 2D visualization projection...")
            topics = self._compute_viz_coordinates(topics, umap_embeddings_5d, comments)
            
            extraction.progress = 0.85
            extraction.current_step = "Naming topics"
            self.db.commit()
            
            # Step 9: Name topics
            print("\n🏷️  Step 9: Naming topics with LLM...")
            config = extraction.config or {}
            llm_model = config.get("llm_model", "openai/gpt-4o-mini")
            print(f"   Model: {llm_model}")
            topics = self._name_topics(topics, llm_model)
            
            extraction.progress = 1.0
            extraction.current_step = "Complete"
            self.db.commit()
            
            # Build result
            num_hierarchical = sum(1 for t in topics if t.get("children"))
            total_subtopics = sum(len(t.get("children", [])) for t in topics)
            
            result = {
                "generated_at": datetime.utcnow().isoformat(),
                "num_comments": len(comments),
                "num_topics": len(topics),
                "num_hierarchical": num_hierarchical,
                "num_subtopics": total_subtopics,
                "outliers": outliers_info,
                "topics": self._clean_topics_for_json(topics),
            }
            
            print("\n" + "=" * 70)
            print("✅ PIPELINE COMPLETE")
            print("=" * 70)
            print(f"   Topics: {len(topics)}")
            print(f"   Hierarchical: {num_hierarchical}")
            print(f"   Sub-topics: {total_subtopics}")
            print(f"   Outliers: {outliers_info['count']} ({outliers_info['percentage']:.1f}%)")
            print(f"   Total comments: {len(comments):,}")
            
            # Save result
            extraction.status = "completed"
            extraction.completed_at = datetime.utcnow()
            extraction.result = result
            extraction.num_topics = len(topics)
            self.db.commit()
            
            return result
            
        except Exception as e:
            extraction.status = "failed"
            extraction.error_message = str(e)
            extraction.completed_at = datetime.utcnow()
            self.db.commit()
            raise
    
    def _load_comments(self, video_ids: list[int]) -> list[str]:
        """Load comment texts from database."""
        comments = self.db.query(Comment.text).filter(
            Comment.video_id.in_(video_ids)
        ).all()
        
        # Filter short comments
        texts = [c.text for c in comments if c.text and len(c.text) > 20]
        return texts
    
    def _run_clustering(
        self,
        comments: list[str],
        embeddings: np.ndarray,
        stopwords: set[str],
        channel_id: int = None,
    ) -> tuple:
        """
        Run robust BERTopic clustering with permissive HDBSCAN + reduction.
        
        Strategy:
        1. Permissive clustering (many topics, few outliers)
        2. Reduce topics to target count
        3. Reassign outliers to nearest clusters
        
        Uses UMAP cache when channel_id is provided to speed up repeated extractions.
        """
        UMAP = _get_umap()
        HDBSCAN = _get_hdbscan()
        BERTopic = _get_bertopic()
        from sklearn.decomposition import PCA
        
        print(f"   Clustering {len(comments):,} comments...")
        
        # Step 0: PCA pre-reduction (384D → 50D) - fast, ~7x speedup for UMAP
        original_dim = embeddings.shape[1]
        pca_target = min(PCA_N_COMPONENTS, len(embeddings) - 1, original_dim)
        print(f"   PCA: {original_dim}D → {pca_target}D (hybrid speedup)...")
        
        pca_model = PCA(n_components=pca_target, random_state=42)
        embeddings_pca = pca_model.fit_transform(embeddings)
        explained_var = sum(pca_model.explained_variance_ratio_) * 100
        print(f"   PCA explained variance: {explained_var:.1f}%")
        
        # Step 1: UMAP on PCA-reduced embeddings (with caching)
        umap_cache = UMAPCache()
        cached_umap = None
        
        if channel_id is not None:
            cached_umap = umap_cache.get_cached_model(channel_id, len(embeddings_pca))
        
        if cached_umap is not None:
            print(f"   UMAP cache HIT - using cached model for channel {channel_id}")
            umap_model = cached_umap
        else:
            print(f"   UMAP cache MISS - fitting new model on {pca_target}D data...")
            umap_model = UMAP(
                n_neighbors=UMAP_N_NEIGHBORS,
                n_components=UMAP_N_COMPONENTS,
                min_dist=UMAP_MIN_DIST,
                metric=UMAP_METRIC,
                random_state=42,
                low_memory=True,
                n_jobs=-1,
            )
        
        # Step 2: Permissive HDBSCAN (accept many micro-topics)
        print(f"   HDBSCAN params: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
              f"min_samples={HDBSCAN_MIN_SAMPLES}, method={HDBSCAN_CLUSTER_SELECTION_METHOD}")
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
            cluster_selection_epsilon=HDBSCAN_CLUSTER_EPSILON,
            metric=HDBSCAN_METRIC,
            prediction_data=True,
        )
        
        vectorizer = CountVectorizer(
            stop_words=list(stopwords),
            min_df=2,
            ngram_range=(1, 2),
        )
        
        # Step 3: Initial BERTopic fit (permissive, many topics expected)
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            min_topic_size=MIN_TOPIC_SIZE,
            calculate_probabilities=True,  # Needed for reduce_outliers
            verbose=False,
        )
        
        topic_labels, probs = topic_model.fit_transform(comments, embeddings_pca)
        
        # Save UMAP model to cache if it was newly fitted
        if cached_umap is None and channel_id is not None:
            # BERTopic fits the UMAP model during fit_transform, retrieve it
            fitted_umap = topic_model.umap_model
            umap_cache.save_model(channel_id, len(embeddings_pca), fitted_umap)
            print(f"   UMAP model cached for channel {channel_id}")
        
        # Count initial results
        initial_topics = len(set(topic_labels) - {-1})
        initial_outliers = sum(1 for l in topic_labels if l == -1)
        initial_outlier_pct = 100 * initial_outliers / len(comments)
        print(f"   Initial: {initial_topics} topics, {initial_outliers} outliers ({initial_outlier_pct:.1f}%)")
        
        # Step 4: Reduce topics to target count
        if initial_topics > self.target_topics:
            print(f"   Reducing topics: {initial_topics} → {self.target_topics}")
            topic_model.reduce_topics(comments, nr_topics=self.target_topics)
            topic_labels = topic_model.topics_
            reduced_topics = len(set(topic_labels) - {-1})
            print(f"   After reduction: {reduced_topics} topics")
        
        # Step 5: Reduce outliers (reassign to nearest cluster)
        outlier_count_before = sum(1 for l in topic_labels if l == -1)
        if outlier_count_before > 0:
            print(f"   Reducing outliers ({OUTLIER_REDUCTION_STRATEGY} strategy)...")
            try:
                new_topics = topic_model.reduce_outliers(
                    comments,
                    topic_labels,
                    strategy=OUTLIER_REDUCTION_STRATEGY,
                    threshold=OUTLIER_REDUCTION_THRESHOLD,
                )
                topic_labels = new_topics
                topic_model.update_topics(comments, topics=new_topics)
            except Exception as e:
                print(f"   Warning: reduce_outliers failed: {e}")
        
        # Final outlier stats
        outlier_indices = [i for i, label in enumerate(topic_labels) if label == -1]
        outlier_count = len(outlier_indices)
        outlier_pct = 100 * outlier_count / len(comments) if comments else 0
        print(f"   Final: {len(set(topic_labels) - {-1})} topics, {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        # Sample random outliers for exploration (max 100)
        sample_size = min(100, outlier_count)
        sampled_indices = random.sample(outlier_indices, sample_size) if outlier_count > 0 else []
        outlier_examples = [comments[i] for i in sampled_indices]
        
        # Build topic structures
        topics = []
        unique_labels = set(topic_labels) - {-1}
        
        for topic_id in sorted(unique_labels):
            mask = np.array(topic_labels) == topic_id
            topic_comments = [comments[i] for i in range(len(comments)) if mask[i]]
            topic_embeddings = embeddings[mask]
            
            # Get top words from BERTopic
            bt_words = topic_model.get_topic(topic_id)
            top_words = [w for w, _ in bt_words[:15]] if bt_words else []
            
            topics.append({
                "id": topic_id,
                "depth": 0,
                "parent_id": None,
                "count": len(topic_comments),
                "comments": topic_comments,
                "embeddings": topic_embeddings,
                "top_words": top_words,
                "indices": np.where(mask)[0].tolist(),
            })
        
        # Get 5D UMAP embeddings for visualization (much faster than 384D → 2D)
        umap_embeddings_5d = topic_model.umap_model.embedding_
        
        # Return topics, model, outlier info, and 5D embeddings
        return topics, topic_model, {
            "count": outlier_count, 
            "percentage": outlier_pct,
            "examples": outlier_examples,
        }, umap_embeddings_5d
    
    def _compute_metrics(
        self,
        topics: list[dict],
        embeddings: np.ndarray,
        comments: list[str],
        topic_model=None,
    ) -> list[dict]:
        """
        Compute DBCV and cluster persistence metrics for each topic.
        
        Uses HDBSCAN's cluster_persistence_ as the per-topic quality metric,
        and global DBCV for overall clustering quality.
        """
        if not topics:
            return topics
        
        # Rebuild labels array
        labels = np.full(len(comments), -1)
        for topic in topics:
            for idx in topic["indices"]:
                labels[idx] = topic["id"]
        
        valid_mask = labels != -1
        if valid_mask.sum() <= 1:
            return topics
        
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Compute global DBCV and per-cluster persistence
        global_dbcv = None
        cluster_persistence = {}
        
        if topic_model is not None:
            try:
                from hdbscan import validity_index
                hdbscan_model = topic_model.hdbscan_model
                
                # Global DBCV score (cast to float64 as hdbscan expects double)
                global_dbcv = validity_index(
                    valid_embeddings.astype(np.float64), 
                    valid_labels
                )
                print(f"   Global DBCV: {global_dbcv:.3f}")
                
                # Get cluster persistence (stability) - this is our per-topic metric
                if hasattr(hdbscan_model, 'cluster_persistence_') and hdbscan_model.cluster_persistence_ is not None:
                    persistence = hdbscan_model.cluster_persistence_
                    unique_ids = sorted(set(valid_labels))
                    for i, topic_id in enumerate(unique_ids):
                        if i < len(persistence):
                            cluster_persistence[topic_id] = float(persistence[i])
                    print(f"   Persistence range: [{min(persistence):.3f}, {max(persistence):.3f}]")
            except Exception as e:
                print(f"   Warning: DBCV/persistence computation failed: {e}")
        
        # Assign metrics to each topic
        for topic in topics:
            # Persistence = cluster stability in HDBSCAN hierarchy
            topic["persistence"] = cluster_persistence.get(topic["id"], None)
            
            # Variance and centroid metrics
            topic["variance"] = float(np.var(topic["embeddings"], axis=0).mean())
            centroid = np.mean(topic["embeddings"], axis=0)
            distances = np.linalg.norm(topic["embeddings"] - centroid, axis=1)
            topic["max_distance"] = float(np.max(distances))
            topic["mean_distance"] = float(np.mean(distances))
        
        return topics
    
    def _extract_semantic_words(
        self,
        topics: list[dict],
        all_comments: list[str],
        stopwords: set,
    ) -> list[dict]:
        """
        Extract semantic top words using centroid similarity + MMR.
        Much better than c-TF-IDF for avoiding stopwords.
        """
        if not topics:
            return topics
        
        print("   Building vocabulary for semantic extraction...")
        
        # Build vocabulary from all comments
        vectorizer = CountVectorizer(
            min_df=SEMANTIC_VOCAB_MIN_DF,
            ngram_range=SEMANTIC_NGRAM_RANGE,
            stop_words=list(stopwords | STOP_WORDS),
        )
        
        try:
            vectorizer.fit(all_comments)
            vocabulary = vectorizer.get_feature_names_out().tolist()
        except ValueError:
            print("   Warning: Could not build vocabulary, using BERTopic words")
            return topics
        
        print(f"   Vocabulary size: {len(vocabulary)} words/ngrams")
        
        # Get vocabulary embeddings (cached)
        embeddings_cache = EmbeddingsCache(self.db, self.embedding_model, EMBEDDING_MODEL_NAME)
        vocab_embeddings = embeddings_cache.get_vocab_embeddings(vocabulary)
        print(f"   Vocabulary embeddings: {vocab_embeddings.shape}")
        
        # OPTIMIZATION: Filter valid topics and batch compute all centroids
        valid_topics = [
            t for t in topics 
            if t.get("embeddings") is not None and len(t.get("embeddings", [])) > 0
        ]
        
        if not valid_topics:
            return topics
        
        # Batch compute all centroids at once (vectorized)
        all_centroids = np.array([
            np.mean(t["embeddings"], axis=0) for t in valid_topics
        ])
        
        # Single large matmul for all topic-vocab similarities (2-4x faster)
        all_similarities = cosine_similarity(all_centroids, vocab_embeddings)
        print(f"   Computed {len(valid_topics)} centroids in batch")
        
        # OPTIMIZATION: Pre-compute vocab similarity matrix for MMR (amortized)
        vocab_sim_matrix = cosine_similarity(vocab_embeddings)
        
        # Extract semantic words for each topic using precomputed data
        for i, topic in enumerate(valid_topics):
            similarities = all_similarities[i]
            
            # Top candidates
            top_candidate_indices = np.argsort(similarities)[-SEMANTIC_CANDIDATES:][::-1]
            candidate_embeddings = vocab_embeddings[top_candidate_indices]
            
            # MMR selection for diversity (using precomputed vocab similarity)
            selected_indices = mmr_selection_fast(
                similarities, 
                candidate_embeddings, 
                top_candidate_indices,
                top_n=SEMANTIC_TOP_N_WORDS, 
                lambda_param=MMR_LAMBDA,
                precomputed_sim_matrix=vocab_sim_matrix,
            )
            
            # Replace top_words with semantic words
            topic["top_words"] = [vocabulary[i] for i in selected_indices]
        
        return topics
    
    def _should_split_topic(self, topic: dict) -> bool:
        """
        Determine if a topic should be split into sub-topics.
        
        Uses mean distance to centroid as quality metric.
        High distance = dispersed cluster = candidate for split.
        """
        # Skip if already at max depth or too small
        if topic.get("depth", 0) >= MAX_DEPTH:
            return False
        if topic["count"] < 30:
            return False
        
        # Check mean_distance (high = dispersed cluster)
        mean_distance = topic.get("mean_distance")
        if mean_distance is not None and mean_distance > self.split_threshold:
            return True
        
        return False
    
    def _split_low_quality_topics(
        self,
        topics: list[dict],
        embeddings: np.ndarray,
        comments: list[str],
    ) -> list[dict]:
        """
        Split topics with high mean_distance into sub-topics.
        
        Uses mean distance to centroid as the quality signal.
        High distance = dispersed cluster = needs sub-clustering.
        """
        UMAP = _get_umap()
        HDBSCAN = _get_hdbscan()
        BERTopic = _get_bertopic()
        
        topics_to_split = [t for t in topics if self._should_split_topic(t)]
        
        if not topics_to_split:
            print("   No topics need splitting!")
            return topics
        
        print(f"   Found {len(topics_to_split)} topics to split (mean_distance > {self.split_threshold})")
        
        for parent_topic in topics_to_split:
            parent_id = parent_topic["id"]
            cluster_comments = parent_topic["comments"]
            cluster_embeddings = parent_topic["embeddings"]
            
            mean_dist = parent_topic.get("mean_distance")
            dist_str = f"dist={mean_dist:.3f}" if mean_dist is not None else "dist=N/A"
            print(f"\n   📌 Splitting topic {parent_id} (count={len(cluster_comments)}, {dist_str})...")
            
            if len(cluster_comments) < 30:
                print(f"      Too few comments ({len(cluster_comments)}), skipping")
                continue
            
            # Sub-clustering with adaptive params
            sub_params = get_adaptive_sub_params(len(cluster_comments))
            print(f"      Sub-clustering params: min_cluster_size={sub_params['min_cluster_size']}, "
                  f"min_topic_size={sub_params['min_topic_size']}")
            
            sub_umap = UMAP(
                n_neighbors=min(15, len(cluster_comments) - 1),
                n_components=min(5, len(cluster_comments) - 2),
                min_dist=0.0,
                metric="cosine",
                random_state=42,
                low_memory=True,
                n_jobs=-1,
            )
            
            sub_hdbscan = HDBSCAN(
                min_cluster_size=sub_params["min_cluster_size"],
                min_samples=max(3, sub_params["min_cluster_size"] // 5),
                metric="euclidean",
                prediction_data=True,
            )
            
            sub_topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=sub_umap,
                hdbscan_model=sub_hdbscan,
                min_topic_size=sub_params["min_topic_size"],
                calculate_probabilities=False,
                verbose=False,
            )
            
            try:
                sub_labels, _ = sub_topic_model.fit_transform(
                    cluster_comments, cluster_embeddings
                )
            except Exception as e:
                print(f"      Error during sub-clustering: {e}")
                continue
            
            # Build sub-topics
            sub_topics = []
            unique_sub = set(sub_labels) - {-1}
            
            # Get persistence for sub-topics from the sub HDBSCAN model
            sub_persistence_dict = {}
            try:
                sub_hdbscan_model = sub_topic_model.hdbscan_model
                if hasattr(sub_hdbscan_model, 'cluster_persistence_') and sub_hdbscan_model.cluster_persistence_ is not None:
                    persistence = sub_hdbscan_model.cluster_persistence_
                    for i, sub_id in enumerate(sorted(unique_sub)):
                        if i < len(persistence):
                            sub_persistence_dict[sub_id] = float(persistence[i])
            except Exception:
                pass  # Persistence not available for sub-topics
            
            for sub_id in sorted(unique_sub):
                mask = np.array(sub_labels) == sub_id
                sub_comments = [cluster_comments[i] for i in range(len(cluster_comments)) if mask[i]]
                sub_embeddings = cluster_embeddings[mask]
                
                bt_words = sub_topic_model.get_topic(sub_id)
                top_words = [w for w, _ in bt_words[:15]] if bt_words else []
                
                # Compute metrics
                if len(sub_comments) > 1:
                    centroid = np.mean(sub_embeddings, axis=0)
                    variance = float(np.var(sub_embeddings, axis=0).mean())
                    distances = np.linalg.norm(sub_embeddings - centroid, axis=1)
                    max_dist = float(np.max(distances))
                    mean_dist = float(np.mean(distances))
                else:
                    variance = 0
                    max_dist = 0
                    mean_dist = 0
                
                sub_topics.append({
                    "id": f"{parent_id}.{sub_id}",
                    "depth": 1,
                    "parent_id": parent_id,
                    "count": len(sub_comments),
                    "persistence": sub_persistence_dict.get(sub_id),
                    "comments": sub_comments,
                    "embeddings": sub_embeddings,
                    "top_words": top_words,
                    "variance": variance,
                    "max_distance": max_dist,
                    "mean_distance": mean_dist,
                })
            
            if sub_topics:
                # Apply semantic word extraction to sub-topics
                sub_topics = self._extract_semantic_words_for_subtopics(sub_topics, cluster_comments)
                
                print(f"      Found {len(sub_topics)} sub-topics")
                for st in sub_topics:
                    pers = st.get("persistence")
                    pers_str = f"pers={pers:.3f}" if pers is not None else "pers=N/A"
                    print(f"         └─ {st['id']}: count={st['count']}, {pers_str}")
                parent_topic["children"] = sub_topics
                parent_topic["is_hierarchical"] = True
            else:
                print("      No sub-topics found")
        
        return topics
    
    def _extract_semantic_words_for_subtopics(
        self,
        sub_topics: list[dict],
        parent_comments: list[str],
    ) -> list[dict]:
        """Extract semantic words for sub-topics using parent vocabulary."""
        if not sub_topics:
            return sub_topics
        
        # Build vocabulary from parent comments
        vectorizer = CountVectorizer(
            min_df=max(2, SEMANTIC_VOCAB_MIN_DF // 2),  # Lower threshold for smaller corpus
            ngram_range=SEMANTIC_NGRAM_RANGE,
            stop_words=list(STOP_WORDS),
        )
        
        try:
            vectorizer.fit(parent_comments)
            vocabulary = vectorizer.get_feature_names_out().tolist()
        except ValueError:
            return sub_topics  # Keep BERTopic words
        
        if len(vocabulary) < 10:
            return sub_topics
        
        # Get vocabulary embeddings
        embeddings_cache = EmbeddingsCache(self.db, self.embedding_model, EMBEDDING_MODEL_NAME)
        vocab_embeddings = embeddings_cache.get_vocab_embeddings(vocabulary)
        
        # Extract semantic words for each sub-topic
        for sub_topic in sub_topics:
            sub_embeddings = sub_topic.get("embeddings")
            if sub_embeddings is None or len(sub_embeddings) == 0:
                continue
            
            centroid = np.mean(sub_embeddings, axis=0).reshape(1, -1)
            similarities = cosine_similarity(centroid, vocab_embeddings)[0]
            
            n_candidates = min(SEMANTIC_CANDIDATES, len(vocabulary))
            top_candidate_indices = np.argsort(similarities)[-n_candidates:][::-1]
            candidate_embeddings = vocab_embeddings[top_candidate_indices]
            
            n_words = min(SEMANTIC_TOP_N_WORDS, len(top_candidate_indices))
            selected_indices = mmr_selection_fast(
                similarities, 
                candidate_embeddings, 
                top_candidate_indices,
                top_n=n_words, 
                lambda_param=MMR_LAMBDA
            )
            
            sub_topic["top_words"] = [vocabulary[i] for i in selected_indices]
        
        return sub_topics
    
    def _compute_viz_coordinates(
        self,
        topics: list[dict],
        embeddings: np.ndarray,
        comments: list[str],
    ) -> list[dict]:
        """
        Compute 2D visualization coordinates for topics using UMAP projection.
        
        Expects 5D UMAP embeddings (from main clustering), not original 384D.
        This makes the 5D → 2D projection very fast.
        
        Adds viz_x, viz_y (centroid position) and viz_spread (cluster spread) 
        to each topic and its children.
        """
        if not topics:
            return topics
        
        UMAP = _get_umap()
        
        # Build label array for all comments
        labels = np.full(len(comments), -1)
        for topic in topics:
            for idx in topic.get("indices", []):
                labels[idx] = topic["id"]
        
        # Only project non-outliers
        valid_mask = labels != -1
        if valid_mask.sum() < 10:
            print("   Not enough points for 2D projection, skipping")
            return topics
        
        valid_embeddings = embeddings[valid_mask]  # Already 5D from main UMAP
        valid_labels = labels[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # UMAP 5D → 2D projection (very fast since input is already 5D)
        print(f"   Projecting {len(valid_embeddings):,} points: {valid_embeddings.shape[1]}D → 2D...")
        
        n_neighbors = min(VIZ_UMAP_N_NEIGHBORS, len(valid_embeddings) - 1)
        
        viz_umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=VIZ_UMAP_MIN_DIST,
            metric=VIZ_UMAP_METRIC,
            random_state=42,
            low_memory=True,
            n_jobs=-1,
        )
        
        viz_coords = viz_umap.fit_transform(valid_embeddings)
        
        # Create mapping from original index to viz coords
        idx_to_viz = {int(valid_indices[i]): viz_coords[i] for i in range(len(valid_indices))}
        
        # Compute centroid and spread for each topic
        for topic in topics:
            topic_indices = topic.get("indices", [])
            topic_coords = np.array([idx_to_viz[i] for i in topic_indices if i in idx_to_viz])
            
            if len(topic_coords) > 0:
                centroid = np.mean(topic_coords, axis=0)
                topic["viz_x"] = float(centroid[0])
                topic["viz_y"] = float(centroid[1])
                
                # Spread = std of distances to centroid
                if len(topic_coords) > 1:
                    distances = np.linalg.norm(topic_coords - centroid, axis=1)
                    topic["viz_spread"] = float(np.std(distances))
                else:
                    topic["viz_spread"] = 0.1
            else:
                # Fallback
                topic["viz_x"] = 0.0
                topic["viz_y"] = 0.0
                topic["viz_spread"] = 0.1
            
            # Compute coordinates for children (subtopics)
            for child in topic.get("children", []):
                child_embeddings = child.get("embeddings")
                if child_embeddings is not None and len(child_embeddings) > 0:
                    # Project child embeddings using the same UMAP (transform)
                    try:
                        child_coords = viz_umap.transform(child_embeddings)
                        child_centroid = np.mean(child_coords, axis=0)
                        child["viz_x"] = float(child_centroid[0])
                        child["viz_y"] = float(child_centroid[1])
                        
                        if len(child_coords) > 1:
                            child_distances = np.linalg.norm(child_coords - child_centroid, axis=1)
                            child["viz_spread"] = float(np.std(child_distances))
                        else:
                            child["viz_spread"] = 0.05
                    except Exception as e:
                        print(f"      Warning: Could not project subtopic {child['id']}: {e}")
                        # Fallback: place near parent with offset
                        child["viz_x"] = topic["viz_x"] + np.random.uniform(-0.5, 0.5)
                        child["viz_y"] = topic["viz_y"] + np.random.uniform(-0.5, 0.5)
                        child["viz_spread"] = 0.05
        
        # Compute global bounds for normalization info
        all_x = [t["viz_x"] for t in topics]
        all_y = [t["viz_y"] for t in topics]
        print(f"   2D projection complete. X range: [{min(all_x):.2f}, {max(all_x):.2f}], "
              f"Y range: [{min(all_y):.2f}, {max(all_y):.2f}]")
        
        return topics

    def _name_topics(self, topics: list[dict], llm_model: str) -> list[dict]:
        """Name topics using LLM."""
        try:
            configure_dspy(llm_model)
        except Exception as e:
            print(f"Warning: Could not configure LLM for naming: {e}")
            # Fallback to keywords-based names
            for topic in topics:
                words = topic.get("top_words", [])[:3]
                topic["generated_name"] = " / ".join(words) if words else f"Topic {topic['id']}"
                
                for child in topic.get("children", []):
                    child_words = child.get("top_words", [])[:3]
                    child["generated_name"] = " / ".join(child_words) if child_words else f"Subtopic {child['id']}"
            return topics
        
        topic_namer = get_topic_namer()
        subtopic_namer = get_subtopic_namer()
        
        print(f"   Naming {len(topics)} top-level topics...")
        
        # Name top-level topics in parallel
        def name_topic(t):
            try:
                return topic_namer(
                    keywords=t.get("top_words", []),
                    comments=t.get("comments", [])[:5],
                )
            except Exception as e:
                print(f"      Error naming topic {t['id']}: {e}")
                words = t.get("top_words", [])[:3]
                return " / ".join(words) if words else f"Topic {t['id']}"
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(name_topic, t): i for i, t in enumerate(topics)}
            
            for future in as_completed(futures):
                idx = futures[future]
                topics[idx]["generated_name"] = future.result()
                topic = topics[idx]
                print(f"      [{idx+1}/{len(topics)}] Topic {topic['id']}: {topic['generated_name']}")
        
        # Name sub-topics with parent context - IN PARALLEL
        all_children = []
        for topic in topics:
            parent_name = topic.get("generated_name", f"Topic {topic['id']}")
            for child in topic.get("children", []):
                child["parent_name"] = parent_name
                all_children.append(child)
        
        if all_children:
            print(f"\n   Naming {len(all_children)} sub-topics in parallel (with parent context)...")
            
            def name_subtopic(child):
                try:
                    return subtopic_namer(
                        parent_name=child["parent_name"],
                        keywords=child.get("top_words", []),
                        comments=child.get("comments", [])[:5],
                    )
                except Exception as e:
                    print(f"      Error naming subtopic {child['id']}: {e}")
                    words = child.get("top_words", [])[:3]
                    return " / ".join(words) if words else f"Subtopic {child['id']}"
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(name_subtopic, c): i for i, c in enumerate(all_children)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    child = all_children[idx]
                    child["generated_name"] = future.result()
                    print(f"      [{idx+1}/{len(all_children)}] {child['id']} (under '{child['parent_name']}'): {child['generated_name']}")
        
        return topics
    
    def _select_representative_comments(
        self, 
        comments: list[str], 
        embeddings: np.ndarray, 
        n: int = 10,
    ) -> list[str]:
        """
        Select representative + diverse comments using centroid + MMR.
        
        Instead of just taking the first N comments, this finds comments that:
        1. Are close to the cluster centroid (representative)
        2. Are diverse from each other (not 5x the same thing reformulated)
        """
        if len(comments) <= n:
            return comments
        
        if embeddings is None or len(embeddings) == 0:
            return comments[:n]
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        
        # Similarity of each comment to centroid
        similarities = cosine_similarity(centroid, embeddings)[0]
        
        # Pre-filter top candidates (2x what we need)
        n_candidates = min(n * 2, len(comments))
        top_indices = np.argsort(similarities)[-n_candidates:][::-1]
        candidate_embeddings = embeddings[top_indices]
        
        # MMR selection for diversity
        selected_indices = mmr_selection_fast(
            similarities,
            candidate_embeddings,
            top_indices,
            top_n=n,
            lambda_param=MMR_LAMBDA,
        )
        
        return [comments[i] for i in selected_indices]
    
    def _to_python_native(self, value):
        """Convert numpy types to Python native types for JSON serialization."""
        if value is None:
            return None
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    
    def _clean_topics_for_json(self, topics: list[dict]) -> list[dict]:
        """Remove non-serializable data and select representative comments."""
        def clean_topic(t):
            # Select representative comments using MMR
            comments = t.get("comments", [])
            embeddings = t.get("embeddings")
            if embeddings is not None and len(comments) > 10:
                example_comments = self._select_representative_comments(
                    comments, embeddings, n=10
                )
            else:
                example_comments = comments[:10]
            
            cleaned = {
                "id": self._to_python_native(t["id"]),
                "depth": self._to_python_native(t.get("depth", 0)),
                "parent_id": self._to_python_native(t.get("parent_id")),
                "parent_name": t.get("parent_name"),
                "generated_name": t.get("generated_name", f"Topic {t['id']}"),
                "count": self._to_python_native(t["count"]),
                "persistence": self._to_python_native(t.get("persistence")),
                "variance": self._to_python_native(t.get("variance")),
                "max_distance": self._to_python_native(t.get("max_distance")),
                "mean_distance": self._to_python_native(t.get("mean_distance")),
                "top_words": t.get("top_words", []),
                "example_comments": example_comments,
                "is_hierarchical": self._to_python_native(t.get("is_hierarchical", False)),
                # 2D visualization coordinates
                "viz_x": self._to_python_native(t.get("viz_x")),
                "viz_y": self._to_python_native(t.get("viz_y")),
                "viz_spread": self._to_python_native(t.get("viz_spread")),
            }
            
            if "children" in t:
                cleaned["children"] = [clean_topic(c) for c in t["children"]]
            
            return cleaned
        
        return [clean_topic(t) for t in topics]
    
    def get_extraction(self, extraction_id: int) -> Optional[Extraction]:
        """Get extraction by ID."""
        return self.db.query(Extraction).get(extraction_id)
    
    def get_extraction_status(self, extraction_id: int) -> dict:
        """Get extraction status."""
        extraction = self.db.query(Extraction).get(extraction_id)
        if not extraction:
            return {"error": "Not found"}
        
        return {
            "id": extraction.id,
            "status": extraction.status,
            "progress": extraction.progress,
            "current_step": extraction.current_step,
            "error_message": extraction.error_message,
            "num_comments": extraction.num_comments,
            "num_topics": extraction.num_topics,
        }

