"""
YouTube API routes.
Handles channel search, video listing, and comment fetching.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db, Channel, Video
from app.services.youtube_service import YouTubeService

router = APIRouter(prefix="/youtube", tags=["youtube"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ChannelSearchRequest(BaseModel):
    """Request to search/add a channel."""
    channel: str  # @handle or URL


class VideoInfo(BaseModel):
    """Video information."""
    id: int
    youtube_id: str
    title: str
    url: str
    has_comments: bool
    comment_count: int


class ChannelResponse(BaseModel):
    """Channel with videos."""
    id: int
    handle: str
    name: str
    channel_id: Optional[str] = None
    description: Optional[str] = None
    subscriber_count: Optional[int] = None
    video_count: int
    videos: list[VideoInfo]


class FetchCommentsRequest(BaseModel):
    """Request to fetch comments for videos."""
    video_ids: list[int]
    max_workers: int = 4


class FetchStatusResponse(BaseModel):
    """Status of comment fetching."""
    active: bool
    channel_id: Optional[int] = None
    channel_name: Optional[str] = None
    videos_total: int
    videos_completed: int
    comments_extracted: int
    current_video: Optional[str] = None


# =============================================================================
# Routes
# =============================================================================

@router.post("/channel", response_model=ChannelResponse)
async def add_or_get_channel(
    request: ChannelSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Add a YouTube channel or get existing one.
    
    If the channel doesn't exist in DB, it will be fetched from YouTube
    and added with all its videos.
    """
    service = YouTubeService(db)
    
    try:
        channel = service.add_channel(request.channel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get videos
    videos = service.get_channel_videos(channel.id)
    
    return ChannelResponse(
        id=channel.id,
        handle=channel.handle,
        name=channel.name,
        channel_id=channel.channel_id,
        description=channel.description,
        subscriber_count=channel.subscriber_count,
        video_count=len(videos),
        videos=[
            VideoInfo(
                id=v.id,
                youtube_id=v.youtube_id,
                title=v.title,
                url=v.url,
                has_comments=v.has_comments,
                comment_count=v.comment_count or 0,
            )
            for v in videos
        ],
    )


@router.get("/channel/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: int,
    db: Session = Depends(get_db),
):
    """Get channel details with videos."""
    channel = db.query(Channel).get(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    videos = db.query(Video).filter(Video.channel_id == channel_id).all()
    
    return ChannelResponse(
        id=channel.id,
        handle=channel.handle,
        name=channel.name,
        channel_id=channel.channel_id,
        description=channel.description,
        subscriber_count=channel.subscriber_count,
        video_count=len(videos),
        videos=[
            VideoInfo(
                id=v.id,
                youtube_id=v.youtube_id,
                title=v.title,
                url=v.url,
                has_comments=v.has_comments,
                comment_count=v.comment_count or 0,
            )
            for v in videos
        ],
    )


@router.get("/channel/{channel_id}/videos")
async def get_channel_videos(
    channel_id: int,
    db: Session = Depends(get_db),
):
    """Get videos for a channel with comment status."""
    channel = db.query(Channel).get(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    videos = db.query(Video).filter(Video.channel_id == channel_id).all()
    
    return {
        "channel_id": channel_id,
        "channel_name": channel.name,
        "total_videos": len(videos),
        "videos_with_comments": sum(1 for v in videos if v.has_comments),
        "videos": [
            {
                "id": v.id,
                "youtube_id": v.youtube_id,
                "title": v.title,
                "url": v.url,
                "has_comments": v.has_comments,
                "comment_count": v.comment_count or 0,
            }
            for v in videos
        ],
    }


@router.post("/videos/fetch-comments")
async def fetch_comments(
    request: FetchCommentsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start fetching comments for selected videos.
    
    This runs in the background. Use /youtube/fetch-status to check progress.
    """
    # Check if already fetching
    status = YouTubeService.get_fetch_status()
    if status["active"]:
        raise HTTPException(
            status_code=409,
            detail="A fetch operation is already in progress",
        )
    
    service = YouTubeService(db)
    
    # Run in background
    def fetch_task():
        # Create new session for background task
        from app.db import SessionLocal
        with SessionLocal() as bg_db:
            bg_service = YouTubeService(bg_db)
            bg_service.fetch_comments_for_videos(
                request.video_ids,
                max_workers=request.max_workers,
            )
    
    background_tasks.add_task(fetch_task)
    
    return {
        "success": True,
        "message": f"Started fetching comments for {len(request.video_ids)} videos",
        "video_count": len(request.video_ids),
    }


@router.get("/fetch-status", response_model=FetchStatusResponse)
async def get_fetch_status():
    """Get current comment fetch status."""
    status = YouTubeService.get_fetch_status()
    return FetchStatusResponse(**status)


@router.post("/fetch-stop")
async def stop_fetch():
    """Stop the current fetch operation."""
    if YouTubeService.request_stop():
        return {"success": True, "message": "Stop requested"}
    else:
        return {"success": False, "message": "No fetch operation in progress"}


@router.get("/channels")
async def list_channels(db: Session = Depends(get_db)):
    """List all channels in database."""
    channels = db.query(Channel).all()
    
    result = []
    for channel in channels:
        video_count = db.query(Video).filter(Video.channel_id == channel.id).count()
        videos_with_comments = db.query(Video).filter(
            Video.channel_id == channel.id,
            Video.has_comments == True,
        ).count()
        
        result.append({
            "id": channel.id,
            "handle": channel.handle,
            "name": channel.name,
            "subscriber_count": channel.subscriber_count,
            "video_count": video_count,
            "videos_with_comments": videos_with_comments,
            "created_at": channel.created_at.isoformat() if channel.created_at else None,
        })
    
    return {"channels": result, "total": len(result)}


@router.delete("/channel/{channel_id}")
async def delete_channel(channel_id: int, db: Session = Depends(get_db)):
    """
    Delete a channel and all associated data (videos, comments, extractions).
    """
    from app.db.models import Comment, Extraction
    
    channel = db.query(Channel).get(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Count what will be deleted
    video_count = db.query(Video).filter(Video.channel_id == channel_id).count()
    video_ids = [v.id for v in db.query(Video).filter(Video.channel_id == channel_id).all()]
    comment_count = db.query(Comment).filter(Comment.video_id.in_(video_ids)).count() if video_ids else 0
    extraction_count = db.query(Extraction).filter(Extraction.channel_id == channel_id).count()
    
    # Delete in order (comments -> videos -> extractions -> channel)
    if video_ids:
        db.query(Comment).filter(Comment.video_id.in_(video_ids)).delete(synchronize_session=False)
    db.query(Video).filter(Video.channel_id == channel_id).delete(synchronize_session=False)
    db.query(Extraction).filter(Extraction.channel_id == channel_id).delete(synchronize_session=False)
    db.delete(channel)
    db.commit()
    
    return {
        "success": True,
        "message": f"Deleted channel {channel.name}",
        "deleted": {
            "videos": video_count,
            "comments": comment_count,
            "extractions": extraction_count,
        }
    }

