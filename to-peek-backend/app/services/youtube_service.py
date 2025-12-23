"""
YouTube service for channel and comment scraping.
Uses yt-dlp for fetching data from YouTube.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import yt_dlp
from sqlalchemy.orm import Session

from app.db.models import Channel, Video, Comment
from app.ml.utils import clean_text


# Global state for tracking fetch progress
_fetch_state = {
    "active": False,
    "channel_id": None,
    "channel_name": None,
    "videos_total": 0,
    "videos_completed": 0,
    "comments_extracted": 0,
    "current_video": None,
    "stop_requested": False,
}
_fetch_lock = threading.Lock()


class YouTubeService:
    """Service for fetching YouTube channel data and comments."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def search_channel(self, channel_input: str) -> dict:
        """
        Search for a channel and return its info.
        
        Args:
            channel_input: Channel handle (@username) or URL
            
        Returns:
            dict with channel info and video list
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "force_generic_extractor": False,
        }
        
        # Build URL
        if not channel_input.startswith("http"):
            if channel_input.startswith("@"):
                url = f"https://www.youtube.com/{channel_input}/videos"
            else:
                url = f"https://www.youtube.com/@{channel_input}/videos"
        else:
            url = channel_input
            if "/videos" not in url:
                url = url.rstrip("/") + "/videos"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
        
        if not result:
            raise ValueError(f"Could not find channel: {channel_input}")
        
        # Extract channel info
        channel_info = {
            "name": result.get("channel", result.get("uploader", "Unknown")),
            "channel_id": result.get("channel_id", result.get("uploader_id", "")),
            "handle": channel_input if channel_input.startswith("@") else f"@{channel_input}",
            "description": result.get("description", ""),
            "subscriber_count": result.get("channel_follower_count"),
        }
        
        # Extract video list
        videos = []
        if "entries" in result:
            for entry in result["entries"]:
                if entry:
                    videos.append({
                        "youtube_id": entry.get("id"),
                        "title": entry.get("title"),
                        "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                    })
        
        return {
            "channel": channel_info,
            "videos": videos,
            "video_count": len(videos),
        }
    
    def add_channel(self, channel_input: str) -> Channel:
        """
        Add a channel to the database (or return existing).
        
        Args:
            channel_input: Channel handle or URL
            
        Returns:
            Channel model instance
        """
        # Normalize handle
        handle = channel_input
        if not handle.startswith("@"):
            if "youtube.com/" in handle:
                # Extract from URL
                if "/@" in handle:
                    handle = "@" + handle.split("/@")[1].split("/")[0]
                else:
                    handle = "@" + handle.split("/")[-1].split("/")[0]
            else:
                handle = f"@{handle}"
        
        # Check if already exists
        existing = self.db.query(Channel).filter(Channel.handle == handle).first()
        if existing:
            return existing
        
        # Fetch channel info
        info = self.search_channel(channel_input)
        channel_data = info["channel"]
        
        # Create channel
        channel = Channel(
            handle=handle,
            name=channel_data["name"],
            channel_id=channel_data["channel_id"],
            description=channel_data.get("description"),
            subscriber_count=channel_data.get("subscriber_count"),
        )
        self.db.add(channel)
        self.db.commit()
        self.db.refresh(channel)
        
        # Add videos
        for video_data in info["videos"]:
            video = Video(
                channel_id=channel.id,
                youtube_id=video_data["youtube_id"],
                title=video_data["title"],
                url=video_data["url"],
                has_comments=False,
            )
            self.db.add(video)
        
        self.db.commit()
        
        return channel
    
    def get_channel_videos(self, channel_id: int) -> list[Video]:
        """Get all videos for a channel."""
        return self.db.query(Video).filter(Video.channel_id == channel_id).all()
    
    def fetch_video_comments(self, video_id: str) -> list[dict]:
        """
        Fetch comments for a single video from YouTube.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of comment dicts
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "getcomments": True,
            "extract_flat": False,
            "extractor_args": {"youtube": {"comment_sort": ["top"], "skip": ["dash", "hls"]}},
        }
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
        
        comments = []
        if result and "comments" in result:
            for comment in result["comments"]:
                comments.append({
                    "author": comment.get("author"),
                    "author_id": comment.get("author_id"),
                    "text": comment.get("text"),
                    "likes": comment.get("like_count", 0),
                    "timestamp": comment.get("timestamp"),
                    "parent_id": comment.get("parent", "root"),
                    "is_reply": comment.get("parent") != "root",
                })
        
        return comments
    
    def fetch_comments_for_videos(
        self,
        video_ids: list[int],
        max_workers: int = 4,
    ) -> dict:
        """
        Fetch comments for multiple videos (by DB ID) in parallel.
        
        Args:
            video_ids: List of Video database IDs
            max_workers: Number of parallel workers
            
        Returns:
            dict with status and stats
        """
        global _fetch_state
        
        # Get videos from DB
        videos = self.db.query(Video).filter(Video.id.in_(video_ids)).all()
        
        # Filter to only videos without comments
        videos_to_fetch = [v for v in videos if not v.has_comments]
        
        if not videos_to_fetch:
            return {
                "success": True,
                "message": "All selected videos already have comments",
                "fetched": 0,
            }
        
        # Extract video data before threading (SQLAlchemy objects are not thread-safe)
        video_data_list = [
            {"id": v.id, "youtube_id": v.youtube_id, "title": v.title}
            for v in videos_to_fetch
        ]
        
        # Update global state
        with _fetch_lock:
            _fetch_state.update({
                "active": True,
                "stop_requested": False,
                "videos_total": len(video_data_list),
                "videos_completed": 0,
                "comments_extracted": 0,
                "current_video": None,
            })
        
        total_comments = 0
        completed = 0
        
        def process_video(video_data: dict):
            """Fetch comments for a single video."""
            try:
                comments = self.fetch_video_comments(video_data["youtube_id"])
                return video_data, comments, None
            except Exception as e:
                return video_data, [], str(e)
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_video, vd): vd for vd in video_data_list}
                
                for future in as_completed(futures):
                    # Check for stop request
                    with _fetch_lock:
                        if _fetch_state["stop_requested"]:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    
                    video_data, comments, error = future.result()
                    completed += 1
                    
                    if not error and comments:
                        # Save comments to DB
                        for comment_data in comments:
                            text = clean_text(comment_data.get("text", ""))
                            if text:
                                comment = Comment(
                                    video_id=video_data["id"],
                                    author=clean_text(comment_data.get("author", "")),
                                    author_id=comment_data.get("author_id"),
                                    text=text,
                                    likes=comment_data.get("likes", 0),
                                    is_reply=comment_data.get("is_reply", False),
                                    parent_id=comment_data.get("parent_id"),
                                    timestamp=comment_data.get("timestamp"),
                                )
                                self.db.add(comment)
                        
                        # Mark video as having comments - fetch fresh from DB
                        video = self.db.query(Video).filter(Video.id == video_data["id"]).first()
                        if video:
                            video.has_comments = True
                            video.comment_count = len(comments)
                        self.db.commit()
                        
                        total_comments += len(comments)
                    
                    # Update state
                    with _fetch_lock:
                        _fetch_state.update({
                            "videos_completed": completed,
                            "comments_extracted": total_comments,
                            "current_video": video_data["title"][:50] if video_data["title"] else "Unknown",
                        })
        
        finally:
            with _fetch_lock:
                _fetch_state["active"] = False
        
        return {
            "success": True,
            "videos_fetched": completed,
            "comments_extracted": total_comments,
            "stopped": _fetch_state.get("stop_requested", False),
        }
    
    @staticmethod
    def get_fetch_status() -> dict:
        """Get current fetch status."""
        with _fetch_lock:
            return _fetch_state.copy()
    
    @staticmethod
    def request_stop():
        """Request stopping the current fetch operation."""
        with _fetch_lock:
            if _fetch_state["active"]:
                _fetch_state["stop_requested"] = True
                return True
        return False

