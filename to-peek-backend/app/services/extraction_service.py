"""
Topic extraction service.
Implements the hierarchical topic extraction pipeline.
"""
import os
import random
import threading
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
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.db.models import Channel, Video, Comment, Extraction
from app.ml import (
    EMBEDDING_MODEL_NAME,
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    HDBSCAN_METRIC,
    MIN_TOPIC_SIZE,
    SEMANTIC_TOP_N_WORDS,
    SEMANTIC_CANDIDATES,
    SEMANTIC_VOCAB_MIN_DF,
    SEMANTIC_NGRAM_RANGE,
    MMR_LAMBDA,
    SILHOUETTE_THRESHOLD,
    STOP_WORDS,
    EmbeddingsCache,
    mmr_selection_fast,
    detect_stopwords,
    get_adaptive_hdbscan_params,
    get_adaptive_sub_params,
)
from app.ml.topic_namer import configure_dspy, get_topic_namer, get_subtopic_namer


# Lazy imports for heavy ML libs
_sentence_transformer = None
_umap = None
_hdbscan = None
_bertopic = None


def _get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _sentence_transformer


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
    4. Calculate silhouette scores
    5. Split low-silhouette clusters
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
            self.embeddings_cache = EmbeddingsCache(self.db, self.embedding_model)
    
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
        
        try:
            # Update status
            extraction.status = "running"
            extraction.started_at = datetime.utcnow()
            extraction.current_step = "Loading comments"
            self.db.commit()
            
            print("=" * 70)
            print("🌳 HIERARCHICAL TOPIC EXTRACTION PIPELINE")
            print("=" * 70)
            
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
            topics, topic_model, outliers_info = self._run_clustering(comments, embeddings, stopwords)
            print(f"   Found {len(topics)} initial topics")
            
            extraction.progress = 0.5
            extraction.current_step = "Computing metrics"
            self.db.commit()
            
            # Step 5: Compute metrics
            print("\n📊 Step 5: Computing silhouette metrics...")
            topics = self._compute_metrics(topics, embeddings, comments)
            
            fourre_tout = sum(1 for t in topics if t.get("silhouette", 1) < SILHOUETTE_THRESHOLD)
            print(f"   Topics with silhouette < {SILHOUETTE_THRESHOLD}: {fourre_tout}")
            for t in topics:
                sil = t.get("silhouette", 0)
                marker = "⚠️ " if sil < SILHOUETTE_THRESHOLD else "✓ "
                print(f"   {marker}Topic {t['id']}: count={t['count']}, sil={sil:.3f}")
            
            extraction.progress = 0.55
            extraction.current_step = "Extracting semantic words"
            self.db.commit()
            
            # Step 6: Extract semantic words (centroid + MMR)
            print("\n🎯 Step 6: Extracting semantic words (centroid + MMR)...")
            topics = self._extract_semantic_words(topics, comments, stopwords)
            
            extraction.progress = 0.65
            extraction.current_step = "Splitting low-quality clusters"
            self.db.commit()
            
            # Step 7: Split low-silhouette clusters
            print(f"\n🔀 Step 7: Splitting low-silhouette topics (threshold={SILHOUETTE_THRESHOLD})...")
            topics = self._split_low_silhouette_topics(topics, embeddings, comments)
            
            extraction.progress = 0.85
            extraction.current_step = "Naming topics"
            self.db.commit()
            
            # Step 8: Name topics
            print("\n🏷️  Step 8: Naming topics with LLM...")
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
    ) -> tuple:
        """Run BERTopic clustering."""
        UMAP = _get_umap()
        HDBSCAN = _get_hdbscan()
        BERTopic = _get_bertopic()
        
        umap_model = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42,
        )
        
        # Adaptive HDBSCAN parameters based on corpus size
        hdbscan_params = get_adaptive_hdbscan_params(len(comments))
        print(f"Adaptive HDBSCAN params for {len(comments)} docs: "
              f"min_cluster_size={hdbscan_params['min_cluster_size']}, "
              f"min_samples={hdbscan_params['min_samples']}")
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_params["min_cluster_size"],
            min_samples=hdbscan_params["min_samples"],
            metric=HDBSCAN_METRIC,
            prediction_data=True,
        )
        
        vectorizer = CountVectorizer(
            stop_words=list(stopwords),
            min_df=2,
            ngram_range=(1, 2),
        )
        
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            min_topic_size=MIN_TOPIC_SIZE,
            calculate_probabilities=False,
            verbose=False,
        )
        
        topic_labels, _ = topic_model.fit_transform(comments, embeddings)
        
        # Collect outliers
        outlier_indices = [i for i, label in enumerate(topic_labels) if label == -1]
        outlier_count = len(outlier_indices)
        outlier_pct = 100 * outlier_count / len(comments) if comments else 0
        print(f"   Outliers: {outlier_count} ({outlier_pct:.1f}%)")
        
        # Sample random outliers for exploration (max 100)
        import random
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
        
        # Return topics, model, and outlier info
        return topics, topic_model, {
            "count": outlier_count, 
            "percentage": outlier_pct,
            "examples": outlier_examples,
        }
    
    def _compute_metrics(
        self,
        topics: list[dict],
        embeddings: np.ndarray,
        comments: list[str],
    ) -> list[dict]:
        """Compute silhouette and other metrics for each topic."""
        if not topics:
            return topics
        
        # Rebuild labels array
        labels = np.full(len(comments), -1)
        for topic in topics:
            for idx in topic["indices"]:
                labels[idx] = topic["id"]
        
        # Compute silhouette for non-outliers
        valid_mask = labels != -1
        if valid_mask.sum() > 1:
            valid_embeddings = embeddings[valid_mask]
            valid_labels = labels[valid_mask]
            
            silhouette_per_sample = silhouette_samples(valid_embeddings, valid_labels)
            
            sample_idx = 0
            for topic in topics:
                topic_size = topic["count"]
                topic_silhouettes = silhouette_per_sample[sample_idx:sample_idx + topic_size]
                
                topic["silhouette"] = float(np.mean(topic_silhouettes))
                topic["variance"] = float(np.var(topic["embeddings"], axis=0).mean())
                
                # Centroid and max distance
                centroid = np.mean(topic["embeddings"], axis=0)
                distances = np.linalg.norm(topic["embeddings"] - centroid, axis=1)
                topic["max_distance"] = float(np.max(distances))
                
                sample_idx += topic_size
        
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
        embeddings_cache = EmbeddingsCache(self.db, self.embedding_model)
        vocab_embeddings = embeddings_cache.get_vocab_embeddings(vocabulary)
        print(f"   Vocabulary embeddings: {vocab_embeddings.shape}")
        
        # Extract semantic words for each topic
        for topic in topics:
            topic_embeddings = topic.get("embeddings")
            if topic_embeddings is None or len(topic_embeddings) == 0:
                continue
            
            # Compute centroid
            centroid = np.mean(topic_embeddings, axis=0).reshape(1, -1)
            
            # Similarity with vocabulary
            similarities = cosine_similarity(centroid, vocab_embeddings)[0]
            
            # Top candidates
            top_candidate_indices = np.argsort(similarities)[-SEMANTIC_CANDIDATES:][::-1]
            candidate_embeddings = vocab_embeddings[top_candidate_indices]
            
            # MMR selection for diversity
            selected_indices = mmr_selection_fast(
                similarities, 
                candidate_embeddings, 
                top_candidate_indices,
                top_n=SEMANTIC_TOP_N_WORDS, 
                lambda_param=MMR_LAMBDA
            )
            
            # Replace top_words with semantic words
            topic["top_words"] = [vocabulary[i] for i in selected_indices]
        
        return topics
    
    def _split_low_silhouette_topics(
        self,
        topics: list[dict],
        embeddings: np.ndarray,
        comments: list[str],
    ) -> list[dict]:
        """Split topics with low silhouette scores."""
        UMAP = _get_umap()
        HDBSCAN = _get_hdbscan()
        BERTopic = _get_bertopic()
        
        topics_to_split = [
            t for t in topics
            if t.get("silhouette", 1) < SILHOUETTE_THRESHOLD
            and t["count"] >= 30
            and t.get("depth", 0) < 1
        ]
        
        if not topics_to_split:
            print("   No topics need splitting!")
            return topics
        
        print(f"   Found {len(topics_to_split)} topics to split")
        
        for parent_topic in topics_to_split:
            parent_id = parent_topic["id"]
            cluster_comments = parent_topic["comments"]
            cluster_embeddings = parent_topic["embeddings"]
            
            print(f"\n   📌 Splitting topic {parent_id} (count={len(cluster_comments)}, sil={parent_topic.get('silhouette', 0):.3f})...")
            
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
            
            # Compute silhouette for sub-topics if we have enough clusters
            sub_silhouette_dict = {}
            valid_sub_mask = np.array(sub_labels) != -1
            unique_valid_labels = set(sub_labels) - {-1}
            if valid_sub_mask.sum() > 1 and len(unique_valid_labels) > 1:
                valid_sub_embeddings = cluster_embeddings[valid_sub_mask]
                valid_sub_labels = np.array(sub_labels)[valid_sub_mask]
                sub_silhouettes = silhouette_samples(valid_sub_embeddings, valid_sub_labels)
                
                # Map silhouette to each sub-topic
                idx = 0
                for sub_id in sorted(unique_valid_labels):
                    count = (np.array(sub_labels) == sub_id).sum()
                    sub_silhouette_dict[sub_id] = float(np.mean(sub_silhouettes[idx:idx + count]))
                    idx += count
            
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
                else:
                    variance = 0
                    max_dist = 0
                
                sub_topics.append({
                    "id": f"{parent_id}.{sub_id}",
                    "depth": 1,
                    "parent_id": parent_id,
                    "count": len(sub_comments),
                    "silhouette": sub_silhouette_dict.get(sub_id),
                    "comments": sub_comments,
                    "embeddings": sub_embeddings,
                    "top_words": top_words,
                    "variance": variance,
                    "max_distance": max_dist,
                })
            
            if sub_topics:
                # Apply semantic word extraction to sub-topics
                sub_topics = self._extract_semantic_words_for_subtopics(sub_topics, cluster_comments)
                
                print(f"      Found {len(sub_topics)} sub-topics")
                for st in sub_topics:
                    sil = st.get("silhouette")
                    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
                    print(f"         └─ {st['id']}: count={st['count']}, sil={sil_str}")
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
        embeddings_cache = EmbeddingsCache(self.db, self.embedding_model)
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
                "id": t["id"],
                "depth": t.get("depth", 0),
                "parent_id": t.get("parent_id"),
                "parent_name": t.get("parent_name"),
                "generated_name": t.get("generated_name", f"Topic {t['id']}"),
                "count": t["count"],
                "silhouette": t.get("silhouette"),
                "variance": t.get("variance"),
                "max_distance": t.get("max_distance"),
                "top_words": t.get("top_words", []),
                "example_comments": example_comments,
                "is_hierarchical": t.get("is_hierarchical", False),
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

