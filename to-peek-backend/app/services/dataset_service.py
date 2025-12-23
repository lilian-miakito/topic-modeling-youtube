"""
Dataset management service.
Handles loading, creating, and querying datasets.
"""
import json
import sys
from pathlib import Path

import polars as pl

from app.core.config import settings
from app.models.datasets import DatasetStats, ChannelStats

# Add modeling lib to path for reusing existing utilities
sys.path.insert(0, str(settings.MODELING_DIR))


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text.strip()


class DatasetService:
    """Service for dataset operations."""
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.datasets_dir = settings.DATASETS_DIR
        self.parquet_file = self.datasets_dir / "comments_full.parquet"
        self.stats_file = self.datasets_dir / "stats.json"
        self.text_file = self.datasets_dir / "comments.txt"
    
    def get_stats(self) -> DatasetStats | None:
        """Get dataset statistics if available."""
        if not self.stats_file.exists():
            return None
        
        with open(self.stats_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert channels dict to proper model
        channels = {
            name: ChannelStats(comments=info["comments"], videos=info["videos"])
            for name, info in data.get("channels", {}).items()
        }
        
        return DatasetStats(
            total_comments=data.get("total_comments", 0),
            unique_channels=data.get("unique_channels", 0),
            unique_videos=data.get("unique_videos", 0),
            replies=data.get("replies", 0),
            top_level_comments=data.get("top_level_comments", 0),
            avg_length=data.get("avg_length", 0),
            channels=channels,
        )
    
    def load_comments(self) -> list[str]:
        """Load all comment texts from the Parquet dataset."""
        if not self.parquet_file.exists():
            return []
        
        df = pl.read_parquet(self.parquet_file)
        return df["text"].to_list()
    
    def load_comments_full(self) -> pl.DataFrame | None:
        """Load full comments dataframe with all metadata."""
        if not self.parquet_file.exists():
            return None
        
        return pl.read_parquet(self.parquet_file)
    
    def sample_comments(self, n: int = 1000, seed: int = 42) -> list[str]:
        """Load a random sample of comments."""
        if not self.parquet_file.exists():
            return []
        
        df = pl.read_parquet(self.parquet_file)
        
        if len(df) <= n:
            return df["text"].to_list()
        
        # Polars sampling
        sampled = df.sample(n=n, seed=seed)
        return sampled["text"].to_list()
    
    def create_dataset(self) -> DatasetStats:
        """
        Create dataset from all comments in data folder.
        Returns statistics about the created dataset.
        """
        all_comments = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all @ChannelName/ directories
        channel_dirs = [
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith("@")
        ]
        
        if not channel_dirs:
            raise FileNotFoundError("No channel directories found")
        
        for channel_dir in channel_dirs:
            comments = self._load_channel_comments(channel_dir)
            all_comments.extend(comments)
        
        if not all_comments:
            raise ValueError("No comments found in any channel")
        
        # Create dataset files
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Text file
        with open(self.text_file, "w", encoding="utf-8", newline="\n") as f:
            for comment in all_comments:
                f.write(comment["text"] + "\n")
        
        # Parquet file
        df = pl.DataFrame(all_comments)
        df.write_parquet(self.parquet_file, compression="zstd")
        
        # Stats
        stats = self._compute_stats(all_comments)
        
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self._stats_to_dict(stats), f, ensure_ascii=False, indent=2)
        
        return stats
    
    def _load_channel_comments(self, channel_dir: Path) -> list[dict]:
        """Load comments from a single channel directory."""
        comments = []
        
        info_file = channel_dir / "info.json"
        if not info_file.exists():
            return comments
        
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        channel_name = info.get("channel_name", channel_dir.name)
        
        videos_dir = channel_dir / "videos"
        if not videos_dir.exists():
            return comments
        
        for video_file in videos_dir.glob("*.json"):
            try:
                with open(video_file, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                video_id = video_data.get("video_id", video_file.stem)
                video_title = clean_text(video_data.get("title", ""))
                
                for comment in video_data.get("comments", []):
                    text = clean_text(comment.get("text", ""))
                    if text:
                        comments.append({
                            "text": text,
                            "channel": channel_name,
                            "video_id": video_id,
                            "video_title": video_title,
                            "author": clean_text(comment.get("author", "")),
                            "likes": comment.get("likes", 0),
                            "is_reply": comment.get("is_reply", False),
                        })
            except Exception:
                continue
        
        return comments
    
    def _compute_stats(self, comments: list[dict]) -> DatasetStats:
        """Compute statistics from comments."""
        channels: dict[str, dict] = {}
        
        for comment in comments:
            channel = comment["channel"]
            if channel not in channels:
                channels[channel] = {"comments": 0, "videos": set()}
            channels[channel]["comments"] += 1
            channels[channel]["videos"].add(comment["video_id"])
        
        # Convert to ChannelStats
        channel_stats = {
            name: ChannelStats(comments=data["comments"], videos=len(data["videos"]))
            for name, data in channels.items()
        }
        
        return DatasetStats(
            total_comments=len(comments),
            unique_channels=len(channels),
            unique_videos=len(set(c["video_id"] for c in comments)),
            replies=sum(1 for c in comments if c["is_reply"]),
            top_level_comments=sum(1 for c in comments if not c["is_reply"]),
            avg_length=sum(len(c["text"]) for c in comments) / len(comments) if comments else 0,
            channels=channel_stats,
        )
    
    def _stats_to_dict(self, stats: DatasetStats) -> dict:
        """Convert stats to JSON-serializable dict."""
        return {
            "total_comments": stats.total_comments,
            "unique_channels": stats.unique_channels,
            "unique_videos": stats.unique_videos,
            "replies": stats.replies,
            "top_level_comments": stats.top_level_comments,
            "avg_length": stats.avg_length,
            "channels": {
                name: {"comments": s.comments, "videos": s.videos}
                for name, s in stats.channels.items()
            },
        }

