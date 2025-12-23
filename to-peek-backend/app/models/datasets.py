"""
Pydantic models for datasets.
"""
from pydantic import BaseModel


class ChannelInfo(BaseModel):
    """Channel information."""
    folder: str
    channel_name: str
    channel_id: str | None = None
    description: str | None = None
    subscriber_count: int | None = None
    video_count: int = 0
    comment_count: int = 0
    last_updated: str | None = None
    size: str = "0 B"


class ChannelStats(BaseModel):
    """Per-channel statistics."""
    comments: int
    videos: int


class DatasetStats(BaseModel):
    """Dataset statistics."""
    total_comments: int
    unique_channels: int
    unique_videos: int
    replies: int
    top_level_comments: int
    avg_length: float
    channels: dict[str, ChannelStats]


class CommentRecord(BaseModel):
    """A single comment with metadata."""
    text: str
    channel: str
    video_id: str
    video_title: str
    author: str
    likes: int = 0
    is_reply: bool = False

