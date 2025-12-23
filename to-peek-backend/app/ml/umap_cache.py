"""
UMAP Model Cache.

Caches fitted UMAP models per channel to avoid re-fitting on repeated extractions.
On cache hit, uses transform() instead of fit_transform(), saving ~10s per extraction.
"""
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from umap import UMAP

logger = logging.getLogger(__name__)

# Cache directory for UMAP models
UMAP_CACHE_DIR = Path("data/umap_cache")


class UMAPCache:
    """
    Cache for fitted UMAP models.
    
    Models are keyed by channel_id and embedding count to ensure
    the cached model is compatible with the current dataset size.
    
    Usage:
        cache = UMAPCache()
        
        # Check for cached model
        cached = cache.get_cached_model(channel_id, n_embeddings)
        
        if cached:
            # Use transform (fast)
            reduced = cached.transform(embeddings)
        else:
            # Fit new model
            model = UMAP(...)
            reduced = model.fit_transform(embeddings)
            cache.save_model(channel_id, n_embeddings, model)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache with optional custom directory."""
        self.cache_dir = cache_dir or UMAP_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, channel_id: int, n_embeddings: int) -> str:
        """Generate cache key from channel and dataset size."""
        return f"channel_{channel_id}_n{n_embeddings}"
    
    def _get_cache_path(self, channel_id: int, n_embeddings: int) -> Path:
        """Get full path to cache file."""
        key = self._get_cache_key(channel_id, n_embeddings)
        return self.cache_dir / f"{key}.pkl"
    
    def get_cached_model(
        self, 
        channel_id: int, 
        n_embeddings: int,
    ) -> Optional["UMAP"]:
        """
        Load cached UMAP model if it exists.
        
        Args:
            channel_id: Channel database ID
            n_embeddings: Number of embeddings in current dataset
            
        Returns:
            Fitted UMAP model or None if not cached
        """
        cache_path = self._get_cache_path(channel_id, n_embeddings)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"UMAP cache HIT: {cache_path.name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load UMAP cache {cache_path}: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None
    
    def save_model(
        self, 
        channel_id: int, 
        n_embeddings: int, 
        model: "UMAP",
    ) -> bool:
        """
        Cache a fitted UMAP model.
        
        Args:
            channel_id: Channel database ID
            n_embeddings: Number of embeddings used for fitting
            model: Fitted UMAP model
            
        Returns:
            True if saved successfully
        """
        cache_path = self._get_cache_path(channel_id, n_embeddings)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"UMAP cache SAVED: {cache_path.name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save UMAP cache {cache_path}: {e}")
            return False
    
    def invalidate_channel(self, channel_id: int) -> int:
        """
        Remove all cached models for a channel.
        
        Useful when channel data changes significantly.
        
        Args:
            channel_id: Channel database ID
            
        Returns:
            Number of cache files removed
        """
        pattern = f"channel_{channel_id}_*.pkl"
        removed = 0
        
        for cache_file in self.cache_dir.glob(pattern):
            try:
                cache_file.unlink()
                removed += 1
            except Exception:
                pass
        
        if removed:
            logger.info(f"Invalidated {removed} UMAP cache(s) for channel {channel_id}")
        
        return removed
    
    def clear_all(self) -> int:
        """
        Clear all cached UMAP models.
        
        Returns:
            Number of cache files removed
        """
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                removed += 1
            except Exception:
                pass
        
        if removed:
            logger.info(f"Cleared {removed} UMAP cache(s)")
        
        return removed
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "num_cached_models": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

