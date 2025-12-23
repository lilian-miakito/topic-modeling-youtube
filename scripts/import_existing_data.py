#!/usr/bin/env python3
"""
Import existing data from data/ folder into the SQLite database.

Usage:
    cd to-peek-backend
    python ../scripts/import_existing_data.py
"""
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "to-peek-backend"))

from app.db.database import SessionLocal, engine, Base
from app.db.models import Channel, Video, Comment


def import_channel(channel_dir: Path, db) -> int:
    """Import a single channel and its videos/comments."""
    info_file = channel_dir / "info.json"
    if not info_file.exists():
        print(f"  ⚠️  No info.json in {channel_dir.name}, skipping")
        return 0
    
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    # Extract handle from folder name
    handle = channel_dir.name  # e.g., "@Hardisk"
    
    # Check if channel already exists
    existing = db.query(Channel).filter(Channel.handle == handle).first()
    if existing:
        print(f"  ⏭️  Channel {handle} already exists (id={existing.id}), skipping")
        return 0
    
    # Create channel
    channel = Channel(
        handle=handle,
        name=info.get("channel_name", handle.replace("@", "")),
        channel_id=info.get("channel_id"),
        description=info.get("description"),
        subscriber_count=info.get("subscriber_count"),
    )
    db.add(channel)
    db.flush()  # Get the ID
    
    print(f"  ✅ Created channel: {handle} (id={channel.id})")
    
    # Import videos
    videos_dir = channel_dir / "videos"
    if not videos_dir.exists():
        print(f"  ⚠️  No videos folder for {handle}")
        return 0
    
    total_comments = 0
    video_files = list(videos_dir.glob("*.json"))
    
    for i, video_file in enumerate(video_files):
        try:
            with open(video_file, "r", encoding="utf-8") as f:
                video_data = json.load(f)
            
            youtube_id = video_data.get("video_id", video_file.stem)
            
            # Check if video already exists
            existing_video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
            if existing_video:
                continue
            
            # Create video
            comments_data = video_data.get("comments", [])
            video = Video(
                channel_id=channel.id,
                youtube_id=youtube_id,
                title=video_data.get("title", "Unknown"),
                url=video_data.get("url", f"https://www.youtube.com/watch?v={youtube_id}"),
                has_comments=len(comments_data) > 0,
                comment_count=len(comments_data),
            )
            db.add(video)
            db.flush()
            
            # Create comments
            for comment_data in comments_data:
                comment = Comment(
                    video_id=video.id,
                    author=comment_data.get("author", "Unknown"),
                    author_id=comment_data.get("author_id"),
                    text=comment_data.get("text", ""),
                    likes=comment_data.get("likes", 0),
                    is_reply=comment_data.get("is_reply", False),
                    parent_id=comment_data.get("parent") if comment_data.get("parent") != "root" else None,
                    timestamp=comment_data.get("timestamp"),
                )
                db.add(comment)
            
            total_comments += len(comments_data)
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"     ... {i + 1}/{len(video_files)} videos processed")
                db.commit()  # Commit in batches
                
        except Exception as e:
            print(f"  ⚠️  Error processing {video_file.name}: {e}")
            continue
    
    db.commit()
    print(f"     📹 {len(video_files)} videos, 💬 {total_comments} comments")
    return total_comments


def main():
    # Find data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"📂 Importing from: {data_dir}")
    print()
    
    # Create tables if needed
    Base.metadata.create_all(bind=engine)
    
    # Get all channel directories
    channel_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("@")])
    
    print(f"Found {len(channel_dirs)} channels to import:")
    for d in channel_dirs:
        print(f"  - {d.name}")
    print()
    
    # Import each channel
    db = SessionLocal()
    total_comments = 0
    
    try:
        for channel_dir in channel_dirs:
            print(f"📺 Processing {channel_dir.name}...")
            total_comments += import_channel(channel_dir, db)
            print()
        
        print("=" * 50)
        print(f"✅ Import complete!")
        print(f"   Total comments imported: {total_comments:,}")
        
        # Summary
        channels = db.query(Channel).count()
        videos = db.query(Video).count()
        comments = db.query(Comment).count()
        print(f"   Database now contains:")
        print(f"   - {channels} channels")
        print(f"   - {videos} videos")
        print(f"   - {comments:,} comments")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()

