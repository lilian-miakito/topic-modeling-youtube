"""
Channel management service.
Handles listing and querying channel data.
"""
import json
import os
from pathlib import Path

from app.core.config import settings
from app.models.datasets import ChannelInfo


class ChannelService:
    """Service for channel operations."""
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
    
    def list_channels(self) -> list[ChannelInfo]:
        """List all channels with their statistics."""
        channels = []
        
        if not self.data_dir.exists():
            return channels
        
        for folder_name in os.listdir(self.data_dir):
            channel_dir = self.data_dir / folder_name
            info_path = channel_dir / "info.json"
            
            if not channel_dir.is_dir():
                continue
            
            channel_info = ChannelInfo(
                folder=folder_name,
                channel_name=folder_name,
            )
            
            # Read info.json if exists
            if info_path.exists():
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    channel_info = ChannelInfo(
                        folder=folder_name,
                        channel_name=data.get("channel_name", folder_name),
                        channel_id=data.get("channel_id"),
                        description=data.get("description"),
                        subscriber_count=data.get("subscriber_count"),
                        video_count=data.get("videos_extracted", 0),
                        comment_count=data.get("total_comments", 0),
                        last_updated=data.get("last_updated"),
                    )
                except Exception:
                    pass
            
            # Calculate folder size
            channel_info.size = self._get_folder_size(channel_dir)
            channels.append(channel_info)
        
        # Sort by last updated
        channels.sort(key=lambda x: x.last_updated or "", reverse=True)
        
        return channels
    
    def get_channel(self, folder: str) -> ChannelInfo | None:
        """Get detailed info for a specific channel."""
        channel_dir = self.data_dir / folder
        
        if not channel_dir.exists():
            return None
        
        info_path = channel_dir / "info.json"
        
        channel_info = ChannelInfo(folder=folder, channel_name=folder)
        
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            channel_info = ChannelInfo(
                folder=folder,
                channel_name=data.get("channel_name", folder),
                channel_id=data.get("channel_id"),
                description=data.get("description"),
                subscriber_count=data.get("subscriber_count"),
                video_count=data.get("videos_extracted", 0),
                comment_count=data.get("total_comments", 0),
                last_updated=data.get("last_updated"),
            )
        
        channel_info.size = self._get_folder_size(channel_dir)
        
        return channel_info
    
    def get_channel_videos(self, folder: str) -> list[dict]:
        """Get all videos with comments for a channel."""
        channel_dir = self.data_dir / folder
        videos_dir = channel_dir / "videos"
        
        if not videos_dir.exists():
            return []
        
        videos = []
        
        for video_file in videos_dir.glob("*.json"):
            try:
                with open(video_file, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                videos.append(video_data)
            except Exception:
                continue
        
        return videos
    
    def _get_folder_size(self, folder: Path) -> str:
        """Calculate folder size and format it."""
        total_size = 0
        
        videos_dir = folder / "videos"
        if videos_dir.exists():
            for f in videos_dir.iterdir():
                if f.is_file():
                    total_size += f.stat().st_size
        
        info_file = folder / "info.json"
        if info_file.exists():
            total_size += info_file.stat().st_size
        
        if total_size < 1024:
            return f"{total_size} B"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.1f} KB"
        else:
            return f"{total_size / (1024 * 1024):.1f} MB"

