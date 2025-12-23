"""
Topic extraction service.
Implements the hierarchical topic extraction pipeline.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
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
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    MIN_TOPIC_SIZE,
    SEMANTIC_TOP_N_WORDS,
    SEMANTIC_CANDIDATES,
    MMR_LAMBDA,
    SILHOUETTE_THRESHOLD,
    STOP_WORDS,
    EmbeddingsCache,
    mmr_selection_fast,
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
            
            # Step 1: Load comments
            comments = self._load_comments(extraction.video_ids)
            if not comments:
                raise ValueError("No comments found for selected videos")
            
            extraction.num_comments = len(comments)
            extraction.progress = 0.1
            extraction.current_step = "Generating embeddings"
            self.db.commit()
            
            # Step 2: Initialize ML and get embeddings
            self._init_ml()
            embeddings = self.embeddings_cache.get_comment_embeddings(comments)
            
            extraction.progress = 0.3
            extraction.current_step = "Clustering"
            self.db.commit()
            
            # Step 3: Run clustering
            topics, topic_model = self._run_clustering(comments, embeddings)
            
            extraction.progress = 0.5
            extraction.current_step = "Computing metrics"
            self.db.commit()
            
            # Step 4: Compute metrics
            topics = self._compute_metrics(topics, embeddings, comments)
            
            extraction.progress = 0.6
            extraction.current_step = "Splitting low-quality clusters"
            self.db.commit()
            
            # Step 5: Split low-silhouette clusters
            topics = self._split_low_silhouette_topics(topics, embeddings, comments)
            
            extraction.progress = 0.8
            extraction.current_step = "Naming topics"
            self.db.commit()
            
            # Step 6: Name topics
            config = extraction.config or {}
            llm_model = config.get("llm_model", "openai/gpt-4o-mini")
            topics = self._name_topics(topics, llm_model)
            
            extraction.progress = 1.0
            extraction.current_step = "Complete"
            self.db.commit()
            
            # Build result
            result = {
                "generated_at": datetime.utcnow().isoformat(),
                "num_comments": len(comments),
                "num_topics": len(topics),
                "num_hierarchical": sum(1 for t in topics if t.get("children")),
                "topics": self._clean_topics_for_json(topics),
            }
            
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
    
    def _run_clustering(self, comments: list[str], embeddings: np.ndarray) -> tuple:
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
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric=HDBSCAN_METRIC,
            prediction_data=True,
        )
        
        vectorizer = CountVectorizer(
            stop_words=list(STOP_WORDS),
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
        
        return topics, topic_model
    
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
            return topics
        
        for parent_topic in topics_to_split:
            parent_id = parent_topic["id"]
            cluster_comments = parent_topic["comments"]
            cluster_embeddings = parent_topic["embeddings"]
            
            if len(cluster_comments) < 30:
                continue
            
            # Sub-clustering with smaller params
            sub_umap = UMAP(
                n_neighbors=min(15, len(cluster_comments) - 1),
                n_components=min(5, len(cluster_comments) - 2),
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            
            sub_hdbscan = HDBSCAN(
                min_cluster_size=max(5, len(cluster_comments) // 10),
                min_samples=3,
                metric="euclidean",
                prediction_data=True,
            )
            
            sub_topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=sub_umap,
                hdbscan_model=sub_hdbscan,
                min_topic_size=max(5, len(cluster_comments) // 15),
                calculate_probabilities=False,
                verbose=False,
            )
            
            try:
                sub_labels, _ = sub_topic_model.fit_transform(
                    cluster_comments, cluster_embeddings
                )
            except Exception:
                continue
            
            # Build sub-topics
            sub_topics = []
            unique_sub = set(sub_labels) - {-1}
            
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
                    "comments": sub_comments,
                    "embeddings": sub_embeddings,
                    "top_words": top_words,
                    "variance": variance,
                    "max_distance": max_dist,
                })
            
            if sub_topics:
                parent_topic["children"] = sub_topics
                parent_topic["is_hierarchical"] = True
        
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
        
        # Name top-level topics in parallel
        def name_topic(t):
            try:
                return topic_namer(
                    keywords=t.get("top_words", []),
                    comments=t.get("comments", [])[:5],
                )
            except Exception:
                words = t.get("top_words", [])[:3]
                return " / ".join(words) if words else f"Topic {t['id']}"
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(name_topic, t): i for i, t in enumerate(topics)}
            
            for future in as_completed(futures):
                idx = futures[future]
                topics[idx]["generated_name"] = future.result()
        
        # Name sub-topics
        for topic in topics:
            parent_name = topic.get("generated_name", f"Topic {topic['id']}")
            
            for child in topic.get("children", []):
                try:
                    child["generated_name"] = subtopic_namer(
                        parent_name=parent_name,
                        keywords=child.get("top_words", []),
                        comments=child.get("comments", [])[:5],
                    )
                    child["parent_name"] = parent_name
                except Exception:
                    words = child.get("top_words", [])[:3]
                    child["generated_name"] = " / ".join(words) if words else f"Subtopic {child['id']}"
                    child["parent_name"] = parent_name
        
        return topics
    
    def _clean_topics_for_json(self, topics: list[dict]) -> list[dict]:
        """Remove non-serializable data from topics."""
        def clean_topic(t):
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
                "example_comments": t.get("comments", [])[:10],
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

