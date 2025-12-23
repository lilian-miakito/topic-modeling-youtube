"""
Dataset API routes.
"""
from fastapi import APIRouter, HTTPException

from app.services.dataset_service import DatasetService
from app.models.datasets import DatasetStats

router = APIRouter(prefix="/datasets", tags=["datasets"])

dataset_service = DatasetService()


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get statistics about the current dataset."""
    stats = dataset_service.get_stats()
    
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset found. Create a dataset first by running create_dataset."
        )
    
    return stats


@router.post("/create", response_model=DatasetStats)
async def create_dataset():
    """
    Create a new dataset from all comments in the data folder.
    This will overwrite any existing dataset.
    """
    try:
        stats = dataset_service.create_dataset()
        return stats
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sample")
async def sample_comments(n: int = 100, seed: int = 42):
    """Get a random sample of comments."""
    comments = dataset_service.sample_comments(n=n, seed=seed)
    
    if not comments:
        raise HTTPException(
            status_code=404,
            detail="No dataset found. Create a dataset first."
        )
    
    return {
        "count": len(comments),
        "sample_size": n,
        "seed": seed,
        "comments": comments,
    }

