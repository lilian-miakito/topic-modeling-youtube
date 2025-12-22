"""
Data loading utilities.
"""
import polars as pl
from .config import DATASETS_DIR


def load_comments() -> list:
    """Load comments from the Parquet dataset."""
    parquet_file = DATASETS_DIR / "comments_full.parquet"
    
    if not parquet_file.exists():
        print(f"Dataset not found: {parquet_file}")
        print("Run create_dataset.py first!")
        return []
    
    df = pl.read_parquet(parquet_file)
    comments = df["text"].to_list()
    
    return comments

