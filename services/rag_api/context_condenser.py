"""
Phase 9: Context Condenser for Intelligent Context Compression

This module implements intelligent sentence selection and context compression to:
- Reduce token usage while maintaining relevant information
- Improve response quality by focusing on most relevant content
- Speed up LLM generation by reducing context size
- Apply semantic clustering and relevance scoring for optimal sentence selection
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class SentenceCandidate:
    """Represents a sentence candidate for context condensation."""
    text: str
    document_id: str
    document_title: str
    sentence_index: int
    document_relevance_score: float
    position_in_document: float  # 0.0 = start, 1.0 = end
    
    # Computed scores
    query_similarity_score: float = 0.0
    relevance_score: float = 0.0
    final_score: float = 0.0
    cluster_id: Optional[int] = None
    is_selected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "sentence_index": self.sentence_index,
            "document_relevance_score": self.document_relevance_score,
            "position_in_document": self.position_in_document,
            "query_similarity_score": self.query_similarity_score,
            "relevance_score": self.relevance_score,
            "final_score": self.final_score,
            "cluster_id": self.cluster_id,
            "is_selected": self.is_selected
        }


class ContextCondenser:
    """
    Intelligent context condenser that selects the most relevant sentences
    from retrieved documents to create a focused context for LLM generation.
    """
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_api_key: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context condenser.
        
        Args:
            embedding_model: Sentence transformer model for semantic similarity
            openai_api_key: OpenAI API key for query embeddings
            config: Configuration dictionary from retrieval_tuning.yaml
        """
        self.embedding_model_name = embedding_model
        self.config = config or {}
        
        # Initialize sentence transformer for local embeddings
        self.sentence_transformer = SentenceTransformer(embedding_model)
        
        # Initialize OpenAI client for query embeddings
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Configuration from retrieval_tuning.yaml
        context_config = self.config.get("context_processing", {}).get("condenser", {})
        self.max_context_length = context_config.get("max_context_length", 4000)
        self.sentence_selection_method = context_config.get("sentence_selection_method", "relevance_score")
        self.relevance_threshold = context_config.get("relevance_threshold", 0.7)
        self.max_sentences = context_config.get("max_sentences", 15)
        self.overlap_reduction = context_config.get("overlap_reduction", True)
        
        # Scoring weights
        scoring_config = self.config.get("context_processing", {}).get("sentence_scoring", {})
        self.query_similarity_weight = scoring_config.get("query_similarity_weight", 0.6)
        self.document_relevance_weight = scoring_config.get("document_relevance_weight", 0.3)
        self.position_weight = scoring_config.get("position_weight", 0.1)
        
        logger.info(f"ContextCondenser initialized with method: {self.sentence_selection_method}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            sentences = sent_tokenize(text)
            # Filter out very short sentences (< 10 characters)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.warning(f"Failed to tokenize sentences with NLTK: {e}, falling back to regex")
            # Fallback to simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.acreate(
                model="text-embedding-3-small",
                input=query
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to get OpenAI embedding for query: {e}")
            # Fallback to sentence transformer
            return self.sentence_transformer.encode([query])[0]
    
    def _extract_sentence_candidates(self,
                                   retrieved_documents: List[Dict[str, Any]]) -> List[SentenceCandidate]:
        """Extract sentence candidates from retrieved documents."""
        candidates = []
        
        for doc in retrieved_documents:
            text = doc.get("content", "")
            doc_id = doc.get("id", "unknown")
            doc_title = doc.get("title", doc.get("metadata", {}).get("title", "Unknown"))
            doc_relevance = doc.get("score", 0.0)
            
            sentences = self._split_into_sentences(text)
            
            for i, sentence in enumerate(sentences):
                # Calculate position in document (0.0 = start, 1.0 = end)
                position = i / len(sentences) if len(sentences) > 1 else 0.0
                
                candidate = SentenceCandidate(
                    text=sentence,
                    document_id=doc_id,
                    document_title=doc_title,
                    sentence_index=i,
                    document_relevance_score=doc_relevance,
                    position_in_document=position
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _calculate_query_similarity_scores(self,
                                               candidates: List[SentenceCandidate],
                                               query: str) -> None:
        """Calculate similarity scores between sentences and query."""
        if not candidates:
            return
        
        # Get query embedding
        query_embedding = await self._get_query_embedding(query)
        
        # Get sentence embeddings
        sentence_texts = [candidate.text for candidate in candidates]
        sentence_embeddings = self.sentence_transformer.encode(sentence_texts)
        
        # Calculate cosine similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # Update candidates with similarity scores
        for candidate, similarity in zip(candidates, similarities):
            candidate.query_similarity_score = float(similarity)
    
    def _calculate_relevance_scores(self, candidates: List[SentenceCandidate]) -> None:
        """Calculate final relevance scores for sentence candidates."""
        for candidate in candidates:
            # Position score (prefer earlier sentences, but not exclusively)
            position_score = 1.0 - (candidate.position_in_document * 0.5)
            
            # Combine scores with weights
            candidate.relevance_score = (
                self.query_similarity_weight * candidate.query_similarity_score +
                self.document_relevance_weight * candidate.document_relevance_score +
                self.position_weight * position_score
            )
            
            candidate.final_score = candidate.relevance_score
    
    def _apply_semantic_clustering(self,
                                 candidates: List[SentenceCandidate],
                                 n_clusters: Optional[int] = None) -> None:
        """Apply semantic clustering to identify diverse sentence groups."""
        if len(candidates) < 2:
            return
        
        # Get sentence embeddings
        sentence_texts = [candidate.text for candidate in candidates]
        sentence_embeddings = self.sentence_transformer.encode(sentence_texts)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(max(2, len(candidates) // 3), 8)
        
        # Perform clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sentence_embeddings)
            
            # Assign cluster IDs to candidates
            for candidate, cluster_id in zip(candidates, cluster_labels):
                candidate.cluster_id = int(cluster_id)
            
            logger.info(f"Applied semantic clustering with {n_clusters} clusters")
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, proceeding without clustering")
    
    def _remove_redundant_sentences(self, candidates: List[SentenceCandidate]) -> List[SentenceCandidate]:
        """Remove redundant sentences based on semantic similarity."""
        if not self.overlap_reduction or len(candidates) <= 1:
            return candidates
        
        # Sort by relevance score (highest first)
        sorted_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        
        selected = []
        sentence_embeddings = []
        
        for candidate in sorted_candidates:
            if not sentence_embeddings:
                # First sentence is always selected
                selected.append(candidate)
                embedding = self.sentence_transformer.encode([candidate.text])[0]
                sentence_embeddings.append(embedding)
                continue
            
            # Calculate similarity with already selected sentences
            candidate_embedding = self.sentence_transformer.encode([candidate.text])[0]
            similarities = cosine_similarity([candidate_embedding], sentence_embeddings)[0]
            
            # Check if sentence is too similar to any selected sentence
            max_similarity = max(similarities)
            if max_similarity < 0.85:  # Threshold for considering sentences similar
                selected.append(candidate)
                sentence_embeddings.append(candidate_embedding)
                
                # Limit the number of sentences
                if len(selected) >= self.max_sentences:
                    break
        
        logger.info(f"Reduced from {len(candidates)} to {len(selected)} sentences after overlap removal")
        return selected
    
    def _select_sentences_by_relevance(self, candidates: List[SentenceCandidate]) -> List[SentenceCandidate]:
        """Select sentences based on relevance scores."""
        # Filter by relevance threshold
        relevant_candidates = [
            c for c in candidates 
            if c.relevance_score >= self.relevance_threshold
        ]
        
        if not relevant_candidates:
            # If no candidates meet threshold, take top candidates anyway
            relevant_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)[:5]
            logger.warning(f"No candidates met relevance threshold {self.relevance_threshold}, using top 5")
        
        # Sort by final score and select top candidates
        relevant_candidates.sort(key=lambda x: x.final_score, reverse=True)
        selected = relevant_candidates[:self.max_sentences]
        
        # Mark as selected
        for candidate in selected:
            candidate.is_selected = True
        
        return selected
    
    def _select_sentences_by_clustering(self, candidates: List[SentenceCandidate]) -> List[SentenceCandidate]:
        """Select sentences using semantic clustering for diversity."""
        if not any(c.cluster_id is not None for c in candidates):
            # Fallback to relevance-based selection
            return self._select_sentences_by_relevance(candidates)
        
        # Group candidates by cluster
        clusters = {}
        for candidate in candidates:
            cluster_id = candidate.cluster_id or 0
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(candidate)
        
        # Select top sentence from each cluster
        selected = []
        sentences_per_cluster = max(1, self.max_sentences // len(clusters))
        
        for cluster_id, cluster_candidates in clusters.items():
            # Sort cluster candidates by relevance
            cluster_candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            # Select top sentences from this cluster
            cluster_selected = cluster_candidates[:sentences_per_cluster]
            selected.extend(cluster_selected)
            
            if len(selected) >= self.max_sentences:
                break
        
        # If we haven't reached max sentences, add more from top candidates
        if len(selected) < self.max_sentences:
            remaining_candidates = [c for c in candidates if c not in selected]
            remaining_candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            needed = self.max_sentences - len(selected)
            selected.extend(remaining_candidates[:needed])
        
        # Mark as selected
        for candidate in selected:
            candidate.is_selected = True
        
        return selected
    
    def _select_sentences_hybrid(self, candidates: List[SentenceCandidate]) -> List[SentenceCandidate]:
        """Hybrid selection combining relevance and clustering."""
        # First apply clustering
        self._apply_semantic_clustering(candidates)
        
        # Select using clustering approach
        cluster_selected = self._select_sentences_by_clustering(candidates)
        
        # If we don't have enough, supplement with pure relevance
        if len(cluster_selected) < self.max_sentences:
            relevance_selected = self._select_sentences_by_relevance(candidates)
            
            # Combine ensuring no duplicates
            combined = list(cluster_selected)
            for candidate in relevance_selected:
                if candidate not in combined and len(combined) < self.max_sentences:
                    combined.append(candidate)
        else:
            combined = cluster_selected
        
        return combined
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _create_condensed_context(self, selected_sentences: List[SentenceCandidate]) -> str:
        """Create the final condensed context from selected sentences."""
        if not selected_sentences:
            return ""
        
        # Sort sentences by document relevance and then by position for logical flow
        selected_sentences.sort(key=lambda x: (x.document_relevance_score, x.sentence_index), reverse=True)
        
        # Group sentences by document
        doc_groups = {}
        for sentence in selected_sentences:
            doc_id = sentence.document_id
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "title": sentence.document_title,
                    "sentences": []
                }
            doc_groups[doc_id]["sentences"].append(sentence)
        
        # Build context with document structure
        context_parts = []
        current_length = 0
        
        for doc_id, doc_info in doc_groups.items():
            doc_title = doc_info["title"]
            sentences = doc_info["sentences"]
            
            # Sort sentences within document by their original position
            sentences.sort(key=lambda x: x.sentence_index)
            
            # Add document header
            doc_header = f"\n--- {doc_title} ---\n"
            
            # Check if adding this document would exceed token limit
            doc_text = doc_header + " ".join(s.text for s in sentences)
            doc_token_count = self._estimate_token_count(doc_text)
            
            if current_length + doc_token_count > self.max_context_length:
                # Add as many sentences as possible
                remaining_tokens = self.max_context_length - current_length - self._estimate_token_count(doc_header)
                
                if remaining_tokens > 50:  # Only add if we have reasonable space
                    context_parts.append(doc_header)
                    current_length += self._estimate_token_count(doc_header)
                    
                    for sentence in sentences:
                        sentence_tokens = self._estimate_token_count(sentence.text)
                        if current_length + sentence_tokens <= self.max_context_length:
                            context_parts.append(sentence.text)
                            current_length += sentence_tokens
                        else:
                            break
                break
            else:
                context_parts.append(doc_header)
                context_parts.extend(s.text for s in sentences)
                current_length += doc_token_count
        
        return " ".join(context_parts)
    
    async def condense_context(self,
                             query: str,
                             retrieved_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main method to condense context from retrieved documents.
        
        Args:
            query: The search query
            retrieved_documents: List of retrieved documents
            
        Returns:
            Dictionary containing:
            - condensed_context: The condensed context string
            - selected_sentences: List of selected sentence information
            - statistics: Performance and selection statistics
        """
        start_time = time.time()
        
        # Extract sentence candidates
        candidates = self._extract_sentence_candidates(retrieved_documents)
        
        if not candidates:
            return {
                "condensed_context": "",
                "selected_sentences": [],
                "statistics": {
                    "total_sentences": 0,
                    "selected_sentences": 0,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "method_used": self.sentence_selection_method,
                    "estimated_tokens_saved": 0
                }
            }
        
        # Calculate query similarity scores
        await self._calculate_query_similarity_scores(candidates, query)
        
        # Calculate relevance scores
        self._calculate_relevance_scores(candidates)
        
        # Select sentences based on configured method
        if self.sentence_selection_method == "relevance_score":
            selected = self._select_sentences_by_relevance(candidates)
        elif self.sentence_selection_method == "semantic_clustering":
            self._apply_semantic_clustering(candidates)
            selected = self._select_sentences_by_clustering(candidates)
        elif self.sentence_selection_method == "hybrid":
            selected = self._select_sentences_hybrid(candidates)
        else:
            logger.warning(f"Unknown selection method: {self.sentence_selection_method}, using relevance_score")
            selected = self._select_sentences_by_relevance(candidates)
        
        # Apply overlap reduction
        if self.overlap_reduction:
            selected = self._remove_redundant_sentences(selected)
        
        # Create final condensed context
        condensed_context = self._create_condensed_context(selected)
        
        # Calculate statistics
        original_text = " ".join(doc.get("content", "") for doc in retrieved_documents)
        original_tokens = self._estimate_token_count(original_text)
        condensed_tokens = self._estimate_token_count(condensed_context)
        
        processing_time = (time.time() - start_time) * 1000
        
        statistics = {
            "total_sentences": len(candidates),
            "selected_sentences": len(selected),
            "processing_time_ms": processing_time,
            "method_used": self.sentence_selection_method,
            "original_estimated_tokens": original_tokens,
            "condensed_estimated_tokens": condensed_tokens,
            "estimated_tokens_saved": original_tokens - condensed_tokens,
            "compression_ratio": condensed_tokens / original_tokens if original_tokens > 0 else 0.0,
            "average_sentence_score": np.mean([s.final_score for s in selected]) if selected else 0.0
        }
        
        logger.info(f"Context condensation completed: {len(selected)}/{len(candidates)} sentences selected in {processing_time:.1f}ms")
        
        return {
            "condensed_context": condensed_context,
            "selected_sentences": [s.to_dict() for s in selected],
            "statistics": statistics
        }


import time  # Add missing import


# Utility function for easy integration
async def condense_retrieval_context(query: str,
                                   retrieved_documents: List[Dict[str, Any]],
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to condense context from retrieved documents.
    
    Args:
        query: The search query
        retrieved_documents: List of retrieved documents from RAG system
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with condensed context and metadata
    """
    condenser = ContextCondenser(config=config)
    return await condenser.condense_context(query, retrieved_documents)
