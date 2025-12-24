"""
Embeddings cache using SQLite.
Provides caching for both comment embeddings and vocabulary embeddings.
"""
import hashlib
import pickle
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import CommentEmbedding, VocabEmbedding

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _hash_text(text: str) -> str:
    """Fast MD5 hash for cache lookup."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes."""
    return pickle.dumps(embedding)


def _deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    return pickle.loads(data)


class EmbeddingsCache:
    """
    Cache for embeddings using SQLite.
    Handles both comment embeddings (by text hash) and vocabulary embeddings (by word).
    Cache is keyed by (text/word, model_name) to support multiple embedding models.
    """
    
    def __init__(self, db: Session, embedding_model: "SentenceTransformer", model_name: str):
        self.db = db
        self.embedding_model = embedding_model
        self.model_name = model_name  # e.g., "BAAI/bge-m3"
    
    def get_comment_embeddings(self, comments: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for comments, using cache for already computed ones.
        
        Args:
            comments: List of comment texts
            show_progress: Whether to show progress bar for new embeddings
            
        Returns:
            numpy array of embeddings (len(comments), embedding_dim)
        """
        if not comments:
            return np.array([])
        
        # Hash all comments
        comment_hashes = [_hash_text(c) for c in comments]
        
        # Load existing from cache (batch to avoid SQLite variable limit)
        BATCH_SIZE = 500
        cache_dict = {}
        for i in range(0, len(comment_hashes), BATCH_SIZE):
            batch_hashes = comment_hashes[i:i + BATCH_SIZE]
            cached = self.db.query(CommentEmbedding).filter(
                CommentEmbedding.text_hash.in_(batch_hashes),
                CommentEmbedding.model_name == self.model_name
            ).all()
            for c in cached:
                cache_dict[c.text_hash] = _deserialize_embedding(c.embedding)
        
        # Find comments not in cache
        new_indices = [i for i, h in enumerate(comment_hashes) if h not in cache_dict]
        new_comments = [comments[i] for i in new_indices]
        new_hashes = [comment_hashes[i] for i in new_indices]
        
        cached_count = len(comments) - len(new_comments)
        print(f"   Comments: {len(comments)} | Cached: {cached_count} | New: {len(new_comments)}")
        
        # Encode only new comments
        if new_comments:
            # Deduplicate new comments (same text can appear multiple times)
            unique_hashes = []
            unique_comments = []
            seen_hashes = set(cache_dict.keys())
            
            for h, c in zip(new_hashes, new_comments):
                if h not in seen_hashes:
                    unique_hashes.append(h)
                    unique_comments.append(c)
                    seen_hashes.add(h)
            
            if unique_comments:
                print(f"   Encoding {len(unique_comments)} unique new comments...")
                new_embeddings = self.embedding_model.encode(
                    unique_comments,
                    batch_size=256,  # With max_seq_length=256, can batch more
                    show_progress_bar=show_progress and len(unique_comments) > 100,
                    normalize_embeddings=True,  # Pre-normalize for cosine similarity
                )
                
                # Add to cache
                for h, emb in zip(unique_hashes, new_embeddings):
                    cache_dict[h] = emb
                    self.db.add(CommentEmbedding(
                        text_hash=h,
                        model_name=self.model_name,
                        embedding=_serialize_embedding(emb)
                    ))
                
                self.db.commit()
                print(f"   Cache updated: {len(cache_dict)} entries total")
            else:
                print("   All new comments were duplicates, already processed")
        else:
            print("   All comments found in cache!")
        
        # Return embeddings in original order
        return np.array([cache_dict[h] for h in comment_hashes])
    
    def get_vocab_embeddings(self, vocabulary: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for vocabulary words, using cache for already computed ones.
        
        Args:
            vocabulary: List of words/n-grams
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (len(vocabulary), embedding_dim)
        """
        if not vocabulary:
            return np.array([])
        
        # Load existing from cache (batch to avoid SQLite variable limit)
        vocab_list = list(vocabulary)
        BATCH_SIZE = 500
        cache_dict = {}
        for i in range(0, len(vocab_list), BATCH_SIZE):
            batch_words = vocab_list[i:i + BATCH_SIZE]
            cached = self.db.query(VocabEmbedding).filter(
                VocabEmbedding.word.in_(batch_words),
                VocabEmbedding.model_name == self.model_name
            ).all()
            for v in cached:
                cache_dict[v.word] = _deserialize_embedding(v.embedding)
        
        # Find words not in cache
        vocab_set = set(vocabulary)
        cached_words = set(cache_dict.keys())
        new_words = list(vocab_set - cached_words)
        
        print(f"   Vocabulary: {len(vocabulary)} | Cached: {len(vocab_set & cached_words)} | New: {len(new_words)}")
        
        # Encode only new words (already deduplicated via set operation)
        if new_words:
            print(f"   Encoding {len(new_words)} new words...")
            new_embeddings = self.embedding_model.encode(
                new_words,
                batch_size=256,  # With max_seq_length=256, can batch more
                show_progress_bar=show_progress and len(new_words) > 100,
                normalize_embeddings=True,
            )
            
            # Add to cache (batch insert)
            for word, emb in zip(new_words, new_embeddings):
                cache_dict[word] = emb
                self.db.add(VocabEmbedding(
                    word=word,
                    model_name=self.model_name,
                    embedding=_serialize_embedding(emb)
                ))
            
            try:
                self.db.commit()
            except Exception:
                # Handle rare duplicates from race conditions
                self.db.rollback()
            print(f"   Cache updated: {len(cache_dict)} words total")
        else:
            print("   All words found in cache!")
        
        # Return embeddings in vocabulary order
        return np.array([cache_dict[word] for word in vocabulary])
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        comment_count = self.db.query(CommentEmbedding).count()
        vocab_count = self.db.query(VocabEmbedding).count()
        
        return {
            "comment_embeddings": comment_count,
            "vocab_embeddings": vocab_count,
        }

