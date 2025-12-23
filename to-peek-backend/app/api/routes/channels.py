"""
Channel API routes.
"""
from fastapi import APIRouter, HTTPException

from app.services.channel_service import ChannelService
from app.models.datasets import ChannelInfo

router = APIRouter(prefix="/channels", tags=["channels"])

channel_service = ChannelService()


@router.get("/", response_model=list[ChannelInfo])
async def list_channels():
    """List all available channels with their statistics."""
    return channel_service.list_channels()


@router.get("/summary")
async def channels_summary():
    """Get summary statistics for all channels."""
    channels = channel_service.list_channels()
    
    total_videos = sum(c.video_count for c in channels)
    total_comments = sum(c.comment_count for c in channels)
    
    return {
        "total_channels": len(channels),
        "total_videos": total_videos,
        "total_comments": total_comments,
        "channels": channels,
    }


@router.get("/{folder}", response_model=ChannelInfo)
async def get_channel(folder: str):
    """Get details for a specific channel."""
    channel = channel_service.get_channel(folder)
    
    if channel is None:
        raise HTTPException(status_code=404, detail=f"Channel '{folder}' not found")
    
    return channel


@router.get("/{folder}/videos")
async def get_channel_videos(folder: str):
    """Get all videos with comments for a channel."""
    channel = channel_service.get_channel(folder)
    
    if channel is None:
        raise HTTPException(status_code=404, detail=f"Channel '{folder}' not found")
    
    videos = channel_service.get_channel_videos(folder)
    
    return {
        "channel": channel,
        "video_count": len(videos),
        "videos": videos,
    }

