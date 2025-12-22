"""
Maximal Marginal Relevance (MMR) for diverse word selection.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def mmr_selection_fast(centroid_sim: np.ndarray, word_embeddings: np.ndarray, 
                        candidate_indices: np.ndarray, top_n: int, 
                        lambda_param: float = 0.7) -> list:
    """
    Fast Maximal Marginal Relevance selection using vectorized operations.
    
    Args:
        centroid_sim: Similarity scores between words and topic centroid (full vocab)
        word_embeddings: Embeddings of candidate words only (pre-filtered)
        candidate_indices: Original indices of candidates in full vocab
        top_n: Number of words to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of selected original indices
    """
    n_candidates = len(candidate_indices)
    if n_candidates == 0:
        return []
    
    # Pre-compute similarity matrix between all candidates (vectorized)
    candidate_sims = cosine_similarity(word_embeddings)  # N x N matrix
    
    # Relevance scores for candidates
    relevance = centroid_sim[candidate_indices]
    
    # Track selected (local indices within candidates)
    selected_local = [np.argmax(relevance)]
    remaining = set(range(n_candidates)) - set(selected_local)
    
    for _ in range(min(top_n - 1, n_candidates - 1)):
        if not remaining:
            break
        
        remaining_list = list(remaining)
        
        # Max similarity to any already selected word (vectorized lookup)
        max_sim_to_selected = candidate_sims[np.ix_(remaining_list, selected_local)].max(axis=1)
        
        # MMR scores
        mmr_scores = lambda_param * relevance[remaining_list] - (1 - lambda_param) * max_sim_to_selected
        
        # Select best
        best_local = remaining_list[np.argmax(mmr_scores)]
        selected_local.append(best_local)
        remaining.remove(best_local)
    
    # Map back to original indices
    return [candidate_indices[i] for i in selected_local]

