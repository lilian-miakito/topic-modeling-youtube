"""
ML Warmup Module.

Pre-loads models and compiles Numba JIT at startup to avoid cold start penalties.
This saves ~10s on the first extraction request.
"""
import logging

import numpy as np

from .config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Global singleton for the embedding model
_embedding_model = None


def get_embedding_model():
    """Get the global SentenceTransformer singleton."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Limit max sequence length for YouTube comments (avg ~150 chars, max ~1000 chars)
        # BGE-M3 defaults to 8192 which wastes GPU memory
        _embedding_model.max_seq_length = 256  # ~1000 chars is plenty
    return _embedding_model


def warmup_ml_components():
    """
    Pre-load models and compile Numba JIT.
    
    Call this at FastAPI startup to avoid cold start penalties:
    - SentenceTransformer model loading: ~5s
    - Numba JIT compilation for UMAP: ~5s
    
    Total savings: ~10s on first extraction request.
    """
    logger.info("Warming up ML components...")
    
    # 1. Load SentenceTransformer singleton
    logger.info("  Loading SentenceTransformer model...")
    model = get_embedding_model()
    logger.info(f"  Model loaded: {EMBEDDING_MODEL_NAME}")
    
    # 2. Warm-up UMAP/Numba JIT with a small dummy dataset
    logger.info("  Compiling UMAP/Numba JIT...")
    from umap import UMAP
    
    dummy_data = np.random.rand(50, 10).astype(np.float32)
    umap_warmup = UMAP(
        n_neighbors=5,
        n_components=2,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
    )
    _ = umap_warmup.fit_transform(dummy_data)
    logger.info("  Numba JIT compiled")
    
    # 3. Warm-up the embedding model with a dummy encode
    logger.info("  Warming up embedding model...")
    _ = model.encode(["warmup text"], show_progress_bar=False)
    logger.info("  Embedding model ready")
    
    logger.info("ML components warmed up successfully!")

