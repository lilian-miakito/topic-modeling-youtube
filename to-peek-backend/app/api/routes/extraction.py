"""
Topic extraction API routes.
Handles starting extractions and getting results.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db, Extraction
from app.services.extraction_service import ExtractionService

router = APIRouter(prefix="/extract", tags=["extraction"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ExtractionConfig(BaseModel):
    """User-configurable extraction parameters."""
    num_topics: int = 15        # Target number of top-level topics (5-50)
    split_threshold: float = 0.75  # Mean distance threshold for splitting (0.6-0.9)


class StartExtractionRequest(BaseModel):
    """Request to start topic extraction."""
    channel_id: int
    video_ids: list[int]
    config: Optional[ExtractionConfig] = None


class ExtractionStatusResponse(BaseModel):
    """Extraction job status."""
    id: int
    status: str
    progress: float
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    num_comments: Optional[int] = None
    num_topics: Optional[int] = None


class TopicInfo(BaseModel):
    """Topic information."""
    id: int | str
    depth: int
    parent_id: Optional[int | str] = None
    parent_name: Optional[str] = None
    generated_name: str
    count: int
    persistence: Optional[float] = None
    variance: Optional[float] = None
    max_distance: Optional[float] = None
    mean_distance: Optional[float] = None
    top_words: list[str] = []
    example_comments: list[str] = []
    is_hierarchical: bool = False
    children: list["TopicInfo"] = []


class OutliersInfo(BaseModel):
    """Outlier statistics."""
    count: int
    percentage: float
    examples: list[str] = []


class ExtractionResultResponse(BaseModel):
    """Full extraction result."""
    id: int
    status: str
    generated_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    num_comments: Optional[int] = None
    num_topics: Optional[int] = None
    num_hierarchical: Optional[int] = None
    num_subtopics: Optional[int] = None
    outliers: Optional[OutliersInfo] = None
    topics: list[TopicInfo] = []


# =============================================================================
# Routes
# =============================================================================

@router.post("/start", response_model=ExtractionStatusResponse)
async def start_extraction(
    request: StartExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start a new topic extraction job.
    
    The extraction runs in the background. Use /extract/status/{id} to check progress.
    """
    service = ExtractionService(db)
    
    # Convert Pydantic config to dict for JSON storage
    config_dict = request.config.model_dump() if request.config else {}
    
    # Create extraction record
    extraction = service.start_extraction(
        channel_id=request.channel_id,
        video_ids=request.video_ids,
        config=config_dict,
    )
    
    # Run extraction in background
    def run_task():
        from app.db import SessionLocal
        with SessionLocal() as bg_db:
            bg_service = ExtractionService(bg_db)
            bg_service.run_extraction(extraction.id)
    
    background_tasks.add_task(run_task)
    
    return ExtractionStatusResponse(
        id=extraction.id,
        status=extraction.status,
        progress=extraction.progress or 0,
        current_step=extraction.current_step,
    )


@router.get("/status/{extraction_id}", response_model=ExtractionStatusResponse)
async def get_extraction_status(
    extraction_id: int,
    db: Session = Depends(get_db),
):
    """Get status of an extraction job."""
    extraction = db.query(Extraction).get(extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    return ExtractionStatusResponse(
        id=extraction.id,
        status=extraction.status,
        progress=extraction.progress or 0,
        current_step=extraction.current_step,
        error_message=extraction.error_message,
        num_comments=extraction.num_comments,
        num_topics=extraction.num_topics,
    )


@router.get("/result/{extraction_id}", response_model=ExtractionResultResponse)
async def get_extraction_result(
    extraction_id: int,
    db: Session = Depends(get_db),
):
    """Get the result of a completed extraction."""
    extraction = db.query(Extraction).get(extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    if extraction.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Extraction not completed. Status: {extraction.status}",
        )
    
    result = extraction.result or {}
    
    # Parse topics recursively
    def parse_topic(t: dict) -> TopicInfo:
        children = [parse_topic(c) for c in t.get("children", [])]
        return TopicInfo(
            id=t.get("id", 0),
            depth=t.get("depth", 0),
            parent_id=t.get("parent_id"),
            parent_name=t.get("parent_name"),
            generated_name=t.get("generated_name", f"Topic {t.get('id')}"),
            count=t.get("count", 0),
            persistence=t.get("persistence"),
            variance=t.get("variance"),
            max_distance=t.get("max_distance"),
            mean_distance=t.get("mean_distance"),
            top_words=t.get("top_words", []),
            example_comments=t.get("example_comments", []),
            is_hierarchical=t.get("is_hierarchical", False),
            children=children,
        )
    
    topics = [parse_topic(t) for t in result.get("topics", [])]
    
    # Parse outliers
    outliers_data = result.get("outliers")
    outliers = None
    if outliers_data:
        outliers = OutliersInfo(
            count=outliers_data.get("count", 0),
            percentage=outliers_data.get("percentage", 0.0),
            examples=outliers_data.get("examples", []),
        )
    
    # Calculate duration
    duration_seconds = None
    if extraction.started_at and extraction.completed_at:
        duration_seconds = (extraction.completed_at - extraction.started_at).total_seconds()
    
    return ExtractionResultResponse(
        id=extraction.id,
        status=extraction.status,
        generated_at=result.get("generated_at"),
        duration_seconds=duration_seconds,
        num_comments=result.get("num_comments"),
        num_topics=result.get("num_topics"),
        num_hierarchical=result.get("num_hierarchical"),
        num_subtopics=result.get("num_subtopics"),
        outliers=outliers,
        topics=topics,
    )


@router.get("/list")
async def list_extractions(
    channel_id: Optional[int] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """List extraction jobs, optionally filtered by channel."""
    query = db.query(Extraction)
    
    if channel_id:
        query = query.filter(Extraction.channel_id == channel_id)
    
    extractions = query.order_by(Extraction.created_at.desc()).limit(limit).all()
    
    return {
        "extractions": [
            {
                "id": e.id,
                "channel_id": e.channel_id,
                "status": e.status,
                "progress": e.progress,
                "num_comments": e.num_comments,
                "num_topics": e.num_topics,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
            }
            for e in extractions
        ],
        "total": len(extractions),
    }

