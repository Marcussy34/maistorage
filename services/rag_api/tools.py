"""
Utility functions for MAI Storage RAG API retrieval system.

This module provides helper functions for text processing, scoring,
ranking algorithms, and retrieval utilities.
"""

import math
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        logger.debug(f"{self.name} completed in {self.elapsed_ms:.2f}ms")


def preprocess_text(text: str, 
                   lowercase: bool = True,
                   remove_extra_whitespace: bool = True,
                   min_length: int = 1) -> str:
    """
    Preprocess text for search and indexing.
    
    Args:
        text: Input text to preprocess
        lowercase: Convert to lowercase
        remove_extra_whitespace: Remove extra whitespace
        min_length: Minimum length filter
        
    Returns:
        Preprocessed text
    """
    if not text or len(text.strip()) < min_length:
        return ""
    
    processed = text
    
    if lowercase:
        processed = processed.lower()
    
    if remove_extra_whitespace:
        # Remove extra whitespace and normalize
        processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed


def normalize_scores(scores: List[float], 
                    method: str = "min_max",
                    safe: bool = True) -> List[float]:
    """
    Normalize a list of scores to [0, 1] range.
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ('min_max', 'z_score', 'robust')
        safe: Return original scores if normalization fails
        
    Returns:
        Normalized scores
    """
    if not scores:
        return scores
    
    try:
        scores_array = np.array(scores)
        
        if method == "min_max":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score == min_score:
                return [0.5] * len(scores)  # All scores are equal
            normalized = (scores_array - min_score) / (max_score - min_score)
            
        elif method == "z_score":
            mean_score = scores_array.mean()
            std_score = scores_array.std()
            if std_score == 0:
                return [0.5] * len(scores)
            normalized = (scores_array - mean_score) / std_score
            # Convert to [0, 1] using sigmoid
            normalized = 1 / (1 + np.exp(-normalized))
            
        elif method == "robust":
            # Use median and MAD for robustness
            median = np.median(scores_array)
            mad = np.median(np.abs(scores_array - median))
            if mad == 0:
                return [0.5] * len(scores)
            normalized = (scores_array - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
            normalized = 1 / (1 + np.exp(-normalized))
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()
        
    except Exception as e:
        logger.warning(f"Score normalization failed: {e}")
        if safe:
            return scores
        raise


def reciprocal_rank_fusion(rankings: List[List[str]], 
                          k: int = 60,
                          weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    Args:
        rankings: List of ranked item lists (each list is a ranking)
        k: RRF parameter (typically 60)
        weights: Optional weights for each ranking (default: equal weights)
        
    Returns:
        List of (item_id, rrf_score) tuples, sorted by score descending
    """
    if not rankings:
        return []
    
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError("Number of weights must match number of rankings")
    
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking, 1):
            rrf_score = weight / (k + rank)
            rrf_scores[item_id] += rrf_score
    
    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items


def maximal_marginal_relevance(query_embedding: np.ndarray,
                              doc_embeddings: np.ndarray,
                              doc_ids: List[str],
                              relevance_scores: List[float],
                              lambda_param: float = 0.5,
                              top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Apply Maximal Marginal Relevance (MMR) for diversity.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embedding matrix (n_docs x embedding_dim)
        doc_ids: Document IDs corresponding to embeddings
        relevance_scores: Initial relevance scores for documents
        lambda_param: Trade-off parameter (0=diversity, 1=relevance)
        top_k: Number of documents to select
        
    Returns:
        List of (doc_id, mmr_score) tuples
    """
    if len(doc_ids) == 0 or top_k <= 0:
        return []
    
    top_k = min(top_k, len(doc_ids))
    
    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Calculate query-document similarities
    query_similarities = np.dot(doc_norms, query_norm)
    
    # Precompute document-document similarities
    doc_similarities = np.dot(doc_norms, doc_norms.T)
    
    selected_indices = []
    remaining_indices = list(range(len(doc_ids)))
    
    for _ in range(top_k):
        if not remaining_indices:
            break
        
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance component
            relevance = lambda_param * relevance_scores[idx]
            
            # Diversity component (max similarity to already selected docs)
            if selected_indices:
                max_similarity = max(doc_similarities[idx][selected_idx] 
                                   for selected_idx in selected_indices)
                diversity = -(1 - lambda_param) * max_similarity
            else:
                diversity = 0
            
            mmr_score = relevance + diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    # Return selected documents with their MMR scores
    result = []
    for i, idx in enumerate(selected_indices):
        doc_id = doc_ids[idx]
        # Use relevance score as the final score (MMR is used for selection only)
        score = relevance_scores[idx]
        result.append((doc_id, score))
    
    return result


@lru_cache(maxsize=1000)
def tokenize_text(text: str, method: str = "simple") -> List[str]:
    """
    Tokenize text for BM25 and other text processing.
    
    Args:
        text: Input text to tokenize
        method: Tokenization method ('simple', 'alpha_only')
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    text = text.lower().strip()
    
    if method == "simple":
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
    elif method == "alpha_only":
        # Only alphabetic tokens
        tokens = re.findall(r'[a-zA-Z]+', text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    
    return tokens


def calculate_bm25_scores(query_tokens: List[str],
                         doc_tokens_list: List[List[str]],
                         k1: float = 1.2,
                         b: float = 0.75) -> List[float]:
    """
    Calculate BM25 scores for documents given a query.
    
    Args:
        query_tokens: Tokenized query
        doc_tokens_list: List of tokenized documents
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        List of BM25 scores for each document
    """
    if not query_tokens or not doc_tokens_list:
        return [0.0] * len(doc_tokens_list)
    
    # Calculate document frequencies and lengths
    N = len(doc_tokens_list)
    doc_lengths = [len(doc_tokens) for doc_tokens in doc_tokens_list]
    avg_doc_length = sum(doc_lengths) / N if N > 0 else 0
    
    # Calculate document frequencies for query terms
    df = defaultdict(int)
    for doc_tokens in doc_tokens_list:
        unique_tokens = set(doc_tokens)
        for token in query_tokens:
            if token in unique_tokens:
                df[token] += 1
    
    scores = []
    
    for doc_tokens in doc_tokens_list:
        doc_length = len(doc_tokens)
        doc_token_counts = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token in doc_token_counts:
                tf = doc_token_counts[token]
                idf = math.log((N - df[token] + 0.5) / (df[token] + 0.5))
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        scores.append(score)
    
    return scores


def combine_scores(scores_dict: Dict[str, List[float]],
                  weights: Optional[Dict[str, float]] = None,
                  method: str = "weighted_sum") -> List[float]:
    """
    Combine multiple score lists using different methods.
    
    Args:
        scores_dict: Dictionary of score name -> score list
        weights: Optional weights for each score type
        method: Combination method ('weighted_sum', 'max', 'rank_fusion')
        
    Returns:
        Combined scores
    """
    if not scores_dict:
        return []
    
    score_names = list(scores_dict.keys())
    score_lists = list(scores_dict.values())
    
    # Validate all score lists have same length
    lengths = [len(scores) for scores in score_lists]
    if len(set(lengths)) > 1:
        raise ValueError("All score lists must have the same length")
    
    if not lengths or lengths[0] == 0:
        return []
    
    n_items = lengths[0]
    
    if weights is None:
        weights = {name: 1.0 for name in score_names}
    
    if method == "weighted_sum":
        # Normalize scores first, then combine
        normalized_scores = {}
        for name, scores in scores_dict.items():
            normalized_scores[name] = normalize_scores(scores)
        
        combined = [0.0] * n_items
        total_weight = sum(weights.values())
        
        for name, scores in normalized_scores.items():
            weight = weights.get(name, 0.0) / total_weight
            for i, score in enumerate(scores):
                combined[i] += weight * score
        
        return combined
    
    elif method == "max":
        # Take maximum score across all methods
        combined = [0.0] * n_items
        for i in range(n_items):
            max_score = max(score_lists[j][i] for j in range(len(score_lists)))
            combined[i] = max_score
        
        return combined
    
    elif method == "rank_fusion":
        # Use reciprocal rank fusion on rankings
        rankings = []
        for scores in score_lists:
            # Convert scores to rankings (higher score = better rank)
            indexed_scores = [(i, score) for i, score in enumerate(scores)]
            sorted_items = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
            ranking = [str(item[0]) for item in sorted_items]
            rankings.append(ranking)
        
        weight_list = [weights.get(name, 1.0) for name in score_names]
        rrf_results = reciprocal_rank_fusion(rankings, weights=weight_list)
        
        # Convert back to score list
        combined = [0.0] * n_items
        for item_id, rrf_score in rrf_results:
            idx = int(item_id)
            combined[idx] = rrf_score
        
        return combined
    
    else:
        raise ValueError(f"Unknown combination method: {method}")


def calculate_metrics(retrieved_ids: List[str],
                     relevant_ids: Set[str],
                     k_values: List[int] = None) -> Dict[str, float]:
    """
    Calculate retrieval metrics (Precision@K, Recall@K, etc.).
    
    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: Set of relevant document IDs
        k_values: List of k values to calculate metrics for
        
    Returns:
        Dictionary of metric name -> value
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    
    metrics = {}
    
    if not retrieved_ids or not relevant_ids:
        # Return zero metrics if no results or no relevant docs
        for k in k_values:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"f1@{k}"] = 0.0
        metrics["map"] = 0.0
        metrics["mrr"] = 0.0
        return metrics
    
    # Calculate precision and recall at different k values
    cumulative_relevant = 0
    average_precisions = []
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            cumulative_relevant += 1
            precision_at_i = cumulative_relevant / (i + 1)
            average_precisions.append(precision_at_i)
        
        # Calculate metrics at specific k values
        rank = i + 1
        if rank in k_values:
            precision_k = cumulative_relevant / rank
            recall_k = cumulative_relevant / len(relevant_ids)
            
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0.0
            
            metrics[f"precision@{rank}"] = precision_k
            metrics[f"recall@{rank}"] = recall_k
            metrics[f"f1@{rank}"] = f1_k
    
    # Mean Average Precision (MAP)
    if average_precisions:
        metrics["map"] = sum(average_precisions) / len(average_precisions)
    else:
        metrics["map"] = 0.0
    
    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr
    
    # Ensure all requested k values have metrics
    for k in k_values:
        if f"precision@{k}" not in metrics:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"f1@{k}"] = 0.0
    
    return metrics


def deduplicate_results(results: List[Tuple[str, float]], 
                       similarity_threshold: float = 0.9) -> List[Tuple[str, float]]:
    """
    Remove near-duplicate results based on text similarity.
    
    Args:
        results: List of (doc_id, score) tuples
        similarity_threshold: Similarity threshold for deduplication
        
    Returns:
        Deduplicated results
    """
    if len(results) <= 1:
        return results
    
    # For now, implement simple exact duplicate removal
    # In practice, you might want to use text similarity
    seen_ids = set()
    deduplicated = []
    
    for doc_id, score in results:
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduplicated.append((doc_id, score))
    
    return deduplicated


def validate_query(query: str, 
                  min_length: int = 1,
                  max_length: int = 1000,
                  allowed_chars: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate a search query.
    
    Args:
        query: Query string to validate
        min_length: Minimum query length
        max_length: Maximum query length
        allowed_chars: Optional regex pattern for allowed characters
        
    Returns:
        (is_valid, error_message) tuple
    """
    if not query:
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    if allowed_chars and not re.match(allowed_chars, query):
        return False, "Query contains invalid characters"
    
    return True, ""


def create_debug_info(dense_results: Optional[List] = None,
                     bm25_results: Optional[List] = None,
                     fusion_method: Optional[str] = None,
                     rerank_method: Optional[str] = None,
                     mmr_applied: bool = False,
                     timing_info: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create debug information dictionary for retrieval response.
    
    Args:
        dense_results: Dense retrieval results info
        bm25_results: BM25 retrieval results info
        fusion_method: Method used for score fusion
        rerank_method: Reranking method applied
        mmr_applied: Whether MMR was applied
        timing_info: Timing information for different stages
        
    Returns:
        Debug information dictionary
    """
    debug = {
        "fusion_method": fusion_method,
        "rerank_method": rerank_method,
        "mmr_applied": mmr_applied,
        "timing": timing_info or {},
    }
    
    if dense_results is not None:
        debug["dense_retrieval"] = {
            "num_results": len(dense_results),
            "score_range": (
                min(r.get("score", 0) for r in dense_results) if dense_results else 0,
                max(r.get("score", 0) for r in dense_results) if dense_results else 0
            )
        }
    
    if bm25_results is not None:
        debug["bm25_retrieval"] = {
            "num_results": len(bm25_results),
            "score_range": (
                min(r.get("score", 0) for r in bm25_results) if bm25_results else 0,
                max(r.get("score", 0) for r in bm25_results) if bm25_results else 0
            )
        }
    
    return debug
