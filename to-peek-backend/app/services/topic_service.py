"""
Topic extraction service.
Wraps BERTopic functionality for the API.
"""
import json
import glob
import os
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.models.topics import (
    TopicExtractionResult,
    TopicExtractionParams,
    TopicInfo,
    HierarchicalTopicResult,
)


class TopicService:
    """Service for topic extraction and management."""
    
    def __init__(self):
        self.datasets_dir = settings.DATASETS_DIR
        self.cache_dir = settings.CACHE_DIR
    
    def get_latest_topics(self) -> TopicExtractionResult | None:
        """Get the most recent topic extraction results."""
        pattern = str(self.datasets_dir / "topics_result_*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to model
        topics = [TopicInfo(**t) for t in data.get("topics", [])]
        
        return TopicExtractionResult(
            generated_at=data.get("generated_at", ""),
            sample_size=data.get("sample_size", 0),
            num_topics=data.get("num_topics", len(topics)),
            params=data.get("params", {}),
            topics=topics,
            filename=Path(latest_file).name,
        )
    
    def get_latest_hierarchical(self) -> HierarchicalTopicResult | None:
        """Get the most recent hierarchical topic results."""
        pattern = str(self.datasets_dir / "topics_hierarchical_*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to model (recursive for children)
        def parse_topic(t: dict) -> TopicInfo:
            children = [parse_topic(c) for c in t.get("children", [])]
            return TopicInfo(
                id=t.get("id", 0),
                count=t.get("count", 0),
                silhouette=t.get("silhouette"),
                variance=t.get("variance"),
                max_distance=t.get("max_distance"),
                top_words_centroid_mmr=t.get("top_words_ctfidf", []),  # Different key in hierarchical
                example_comments_original=t.get("example_comments", []),
                generated_name=t.get("generated_name"),
                depth=t.get("depth", 0),
                parent_id=t.get("parent_id"),
                parent_name=t.get("parent_name"),
                is_hierarchical=t.get("is_hierarchical", False),
                children=children,
            )
        
        topics = [parse_topic(t) for t in data.get("topics", [])]
        
        return HierarchicalTopicResult(
            generated_at=data.get("generated_at", ""),
            pipeline=data.get("pipeline", "hierarchical"),
            config=data.get("config", {}),
            num_topics=data.get("num_topics", len(topics)),
            num_hierarchical=data.get("num_hierarchical", 0),
            num_subtopics=data.get("num_subtopics", 0),
            topics=topics,
            filename=Path(latest_file).name,
        )
    
    def list_topic_files(self) -> list[dict]:
        """List all topic result files."""
        files = []
        
        # Regular topics
        for pattern in ["topics_result_*.json", "topics_hierarchical_*.json", "topics_named_*.json"]:
            for filepath in glob.glob(str(self.datasets_dir / pattern)):
                path = Path(filepath)
                files.append({
                    "name": path.name,
                    "type": "hierarchical" if "hierarchical" in path.name else "flat",
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "size": path.stat().st_size,
                })
        
        # Sort by modification time
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return files
    
    def get_stopwords(self) -> dict | None:
        """Get detected stopwords."""
        stopwords_file = self.cache_dir / "detected_stopwords.json"
        
        if not stopwords_file.exists():
            return None
        
        with open(stopwords_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_visualizations(self) -> list[dict]:
        """List available visualization files."""
        viz_dir = settings.MODELING_DIR / "visualizations"
        
        if not viz_dir.exists():
            return []
        
        files = []
        for f in viz_dir.iterdir():
            if f.suffix == ".html":
                files.append({
                    "name": f.name,
                    "type": f.stem.split("_")[0],
                    "path": f"/visualizations/{f.name}",
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return files
    
    # TODO: Implement extract_topics() method that runs the actual extraction
    # This will be a long-running operation, possibly with background tasks

