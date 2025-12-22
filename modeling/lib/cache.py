"""
Caching utilities for embeddings.
"""
import hashlib
import numpy as np
import polars as pl

from .config import CACHE_DIR, VOCAB_CACHE_FILE, COMMENTS_CACHE_FILE


def _hash_text(text: str) -> str:
    """Fast hash for cache lookup."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_comments_embeddings_cached(comments: list, embedding_model) -> np.ndarray:
    """
    Get embeddings for comments, using cache for already computed ones.
    Uses MD5 hash of text as cache key.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load existing cache
    cache_dict = {}
    if COMMENTS_CACHE_FILE.exists():
        cache_df = pl.read_parquet(COMMENTS_CACHE_FILE)
        for row in cache_df.iter_rows(named=True):
            cache_dict[row['hash']] = np.array(row['embedding'])
        print(f"   Comments cache loaded: {len(cache_dict)} entries")
    
    # Hash all comments
    comment_hashes = [_hash_text(c) for c in comments]
    
    # Find comments not in cache
    new_indices = [i for i, h in enumerate(comment_hashes) if h not in cache_dict]
    new_comments = [comments[i] for i in new_indices]
    new_hashes = [comment_hashes[i] for i in new_indices]
    
    cached_count = len(comments) - len(new_comments)
    print(f"   Comments: {len(comments)} | Cached: {cached_count} | New: {len(new_comments)}")
    
    # Encode only new comments
    if new_comments:
        print(f"   Encoding {len(new_comments)} new comments...")
        new_embeddings = embedding_model.encode(
            new_comments,
            batch_size=32,
            show_progress_bar=len(new_comments) > 100
        )
        
        # Add to cache dict
        for h, emb in zip(new_hashes, new_embeddings):
            cache_dict[h] = emb
        
        # Save updated cache
        cache_data = {
            'hash': list(cache_dict.keys()),
            'embedding': [emb.tolist() for emb in cache_dict.values()]
        }
        cache_df = pl.DataFrame(cache_data)
        cache_df.write_parquet(COMMENTS_CACHE_FILE)
        print(f"   Comments cache updated: {len(cache_dict)} entries total")
    else:
        print("   All comments found in cache!")
    
    # Return embeddings in comments order
    return np.array([cache_dict[h] for h in comment_hashes])


def get_vocab_embeddings_cached(vocabulary: list, embedding_model) -> np.ndarray:
    """
    Get embeddings for vocabulary words, using cache for already computed ones.
    Only encodes new words and updates the cache.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Load existing cache
    cache_dict = {}
    if VOCAB_CACHE_FILE.exists():
        cache_df = pl.read_parquet(VOCAB_CACHE_FILE)
        for row in cache_df.iter_rows(named=True):
            cache_dict[row['word']] = np.array(row['embedding'])
        print(f"   Cache loaded: {len(cache_dict)} words")
    
    # Find words not in cache
    vocab_set = set(vocabulary)
    cached_words = set(cache_dict.keys())
    new_words = list(vocab_set - cached_words)
    
    print(f"   Vocabulary: {len(vocabulary)} | Cached: {len(vocab_set & cached_words)} | New: {len(new_words)}")
    
    # Encode only new words
    if new_words:
        print(f"   Encoding {len(new_words)} new words...")
        new_embeddings = embedding_model.encode(
            new_words,
            batch_size=128,
            show_progress_bar=len(new_words) > 100
        )
        
        # Add to cache dict
        for word, emb in zip(new_words, new_embeddings):
            cache_dict[word] = emb
        
        # Save updated cache
        cache_data = {
            'word': list(cache_dict.keys()),
            'embedding': [emb.tolist() for emb in cache_dict.values()]
        }
        cache_df = pl.DataFrame(cache_data)
        cache_df.write_parquet(VOCAB_CACHE_FILE)
        print(f"   Cache updated: {len(cache_dict)} words total")
    else:
        print("   All words found in cache!")
    
    # Return embeddings in vocabulary order
    return np.array([cache_dict[word] for word in vocabulary])

