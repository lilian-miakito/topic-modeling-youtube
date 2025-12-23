"""
Topic extraction API routes.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services.topic_service import TopicService
from app.models.topics import TopicExtractionResult, HierarchicalTopicResult

router = APIRouter(prefix="/topics", tags=["topics"])

topic_service = TopicService()


@router.get("/", response_model=TopicExtractionResult)
async def get_latest_topics():
    """Get the most recent topic extraction results."""
    result = topic_service.get_latest_topics()
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No topic results found. Run topic extraction first."
        )
    
    return result


@router.get("/hierarchical", response_model=HierarchicalTopicResult)
async def get_latest_hierarchical():
    """Get the most recent hierarchical topic extraction results."""
    result = topic_service.get_latest_hierarchical()
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No hierarchical topic results found. Run hierarchical pipeline first."
        )
    
    return result


@router.get("/files")
async def list_topic_files():
    """List all available topic result files."""
    return topic_service.list_topic_files()


@router.get("/stopwords")
async def get_stopwords():
    """Get detected stopwords."""
    stopwords = topic_service.get_stopwords()
    
    if stopwords is None:
        raise HTTPException(
            status_code=404,
            detail="No stopwords detected. Run detect_stopwords first."
        )
    
    return stopwords


@router.get("/visualizations")
async def list_visualizations():
    """List available visualization files."""
    return topic_service.list_visualizations()


@router.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """Serve a visualization HTML file."""
    viz_dir = settings.MODELING_DIR / "visualizations"
    filepath = viz_dir / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Visualization '{filename}' not found")
    
    return FileResponse(filepath, media_type="text/html")


# TODO: Add endpoints for:
# - POST /extract - Start topic extraction (background task)
# - GET /extract/status - Get extraction status
# - POST /name - Name topics using LLM

