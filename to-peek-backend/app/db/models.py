"""
SQLAlchemy ORM models for To-Peek.

Tables:
- channels: YouTube channels
- videos: Videos from channels
- comments: Comments on videos
- comment_embeddings: Cached embeddings for comments (by text hash)
- vocab_embeddings: Cached embeddings for vocabulary words/n-grams
- extractions: Topic extraction jobs and results
"""
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    LargeBinary,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship

from .database import Base


class Channel(Base):
    """YouTube channel."""
    __tablename__ = "channels"
    
    id = Column(Integer, primary_key=True, index=True)
    handle = Column(String(100), unique=True, index=True)  # @username
    name = Column(String(255))
    channel_id = Column(String(50))  # YouTube channel ID
    description = Column(Text, nullable=True)
    subscriber_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")
    extractions = relationship("Extraction", back_populates="channel", cascade="all, delete-orphan")


class Video(Base):
    """YouTube video."""
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    youtube_id = Column(String(20), unique=True, index=True)  # Video ID (e.g., "dQw4w9WgXcQ")
    title = Column(String(500))
    url = Column(String(255))
    has_comments = Column(Boolean, default=False)  # True if comments have been fetched
    comment_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    channel = relationship("Channel", back_populates="videos")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")


class Comment(Base):
    """YouTube comment."""
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    author = Column(String(255))
    author_id = Column(String(50), nullable=True)
    text = Column(Text)
    likes = Column(Integer, default=0)
    is_reply = Column(Boolean, default=False)
    parent_id = Column(String(50), nullable=True)  # YouTube comment ID of parent
    timestamp = Column(Integer, nullable=True)  # Unix timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", back_populates="comments")
    
    # Index for faster text lookups
    __table_args__ = (
        Index("ix_comments_video_text", "video_id", "text"),
    )


class CommentEmbedding(Base):
    """
    Cache for comment text embeddings.
    Uses MD5 hash of text + model name as key for deduplication.
    """
    __tablename__ = "comment_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String(32), index=True)  # MD5 hash
    model_name = Column(String(100), index=True)  # Embedding model name
    embedding = Column(LargeBinary)  # numpy array as bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_comment_embeddings_hash_model", "text_hash", "model_name", unique=True),
    )


class VocabEmbedding(Base):
    """
    Cache for vocabulary word/n-gram embeddings.
    Uses the word + model name as key.
    """
    __tablename__ = "vocab_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String(200), index=True)  # Word or n-gram
    model_name = Column(String(100), index=True)  # Embedding model name
    embedding = Column(LargeBinary)  # numpy array as bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_vocab_embeddings_word_model", "word", "model_name", unique=True),
    )


class Extraction(Base):
    """
    Topic extraction job.
    Stores configuration, status, and results.
    """
    __tablename__ = "extractions"
    
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    
    # Job status
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    current_step = Column(String(100), nullable=True)  # Current step description
    error_message = Column(Text, nullable=True)
    
    # Configuration
    config = Column(JSON, default=dict)  # Extraction parameters
    video_ids = Column(JSON, default=list)  # List of video IDs to include
    
    # Results
    result = Column(JSON, nullable=True)  # Hierarchical topics result
    num_topics = Column(Integer, nullable=True)
    num_comments = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    channel = relationship("Channel", back_populates="extractions")

