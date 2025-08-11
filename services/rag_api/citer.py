"""
Sentence-level citation engine for Phase 6 of MAI Storage RAG system.

This module implements post-hoc attribution that maps individual sentences in 
generated responses to supporting text spans in source documents using 
semantic similarity and confidence scoring.

Key features:
- Sentence extraction and embedding
- Cosine similarity-based attribution  
- Confidence scoring and threshold-based warnings
- Text span identification for precise attribution
- Optional LLM-based sentence rephrasing for better alignment
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# NLP dependencies for sentence processing
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Local imports
from models import (
    SentenceCitation, SentenceAttributionResult, EnhancedCitation,
    TextSpan, CitationEngineConfig, RetrievalResult
)
from llm_client import LLMClient, LLMConfig
from retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class SentenceCitationEngine:
    """
    Core engine for sentence-level attribution and citation generation.
    
    This class takes generated responses and retrieval results, then performs
    post-hoc attribution to map each sentence to supporting source material
    with confidence scoring.
    """
    
    def __init__(self, 
                 retriever: HybridRetriever,
                 llm_client: Optional[LLMClient] = None,
                 config: Optional[CitationEngineConfig] = None):
        """
        Initialize the citation engine.
        
        Args:
            retriever: HybridRetriever instance for embedding generation
            llm_client: Optional LLM client for sentence rephrasing
            config: Configuration for attribution thresholds and methods
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config or CitationEngineConfig()
        
        # Performance tracking
        self.stats = {
            "total_attributions": 0,
            "total_sentences_processed": 0,
            "total_time_ms": 0.0,
            "average_confidence": 0.0
        }
        
        logger.info(f"SentenceCitationEngine initialized with config: {self.config.primary_attribution_method}")
    
    async def generate_sentence_citations(self, 
                                        response_text: str,
                                        retrieval_results: List[RetrievalResult]) -> SentenceAttributionResult:
        """
        Generate sentence-level citations for a response.
        
        Args:
            response_text: The generated response text to cite
            retrieval_results: Source documents from retrieval
            
        Returns:
            SentenceAttributionResult with complete attribution information
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract sentences from response
            sentences = self._extract_sentences(response_text)
            if not sentences:
                return self._create_empty_result(response_text, "No sentences extracted")
            
            logger.info(f"Processing {len(sentences)} sentences for attribution")
            
            # Step 2: Prepare source materials
            source_materials = self._prepare_source_materials(retrieval_results)
            if not source_materials:
                return self._create_empty_result(response_text, "No source materials available")
            
            # Step 3: Generate embeddings for sentences
            sentence_embeddings = await self._generate_sentence_embeddings(sentences)
            
            # Step 4: Perform attribution for each sentence
            sentence_citations = []
            for i, (sentence, sentence_embedding) in enumerate(zip(sentences, sentence_embeddings)):
                citation = await self._attribute_sentence(
                    sentence=sentence,
                    sentence_index=i,
                    sentence_embedding=sentence_embedding,
                    source_materials=source_materials
                )
                sentence_citations.append(citation)
            
            # Step 5: Calculate overall metrics
            attribution_time = (time.time() - start_time) * 1000
            result = self._create_attribution_result(
                response_text=response_text,
                sentences=sentences,
                sentence_citations=sentence_citations,
                attribution_time_ms=attribution_time
            )
            
            # Update stats
            self._update_stats(len(sentences), attribution_time, result.overall_confidence)
            
            logger.info(f"Attribution completed: {result.attribution_coverage:.2%} coverage, "
                       f"{result.overall_confidence:.3f} confidence, {attribution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Attribution failed: {e}")
            return self._create_empty_result(response_text, f"Attribution error: {str(e)}")
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        try:
            # Use NLTK for robust sentence tokenization
            sentences = sent_tokenize(text.strip())
            
            # Filter sentences by minimum length
            filtered_sentences = [
                s.strip() for s in sentences 
                if len(s.strip()) >= self.config.min_sentence_length
            ]
            
            # Limit to maximum sentences
            if len(filtered_sentences) > self.config.max_sentences_per_response:
                logger.warning(f"Truncating to {self.config.max_sentences_per_response} sentences")
                filtered_sentences = filtered_sentences[:self.config.max_sentences_per_response]
            
            return filtered_sentences
            
        except Exception as e:
            logger.error(f"Sentence extraction failed: {e}")
            return []
    
    def _prepare_source_materials(self, retrieval_results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Prepare source materials for attribution."""
        source_materials = []
        
        for result in retrieval_results:
            # Extract document text and metadata
            doc = result.document
            material = {
                "document_id": doc.id,
                "doc_name": doc.doc_name,
                "chunk_index": doc.chunk_index,
                "text": doc.text,
                "score": result.final_score or result.hybrid_score or result.dense_score or 0.0,
                "char_count": len(doc.text)
            }
            source_materials.append(material)
        
        # Sort by relevance score (highest first)
        source_materials.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Prepared {len(source_materials)} source materials for attribution")
        return source_materials
    
    async def _generate_sentence_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of sentences."""
        embeddings = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            try:
                # Generate embeddings using the retriever's OpenAI client
                batch_embeddings = []
                for sentence in batch:
                    embedding = await self.retriever.embed_query(sentence)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                # Add zero embeddings as fallback
                batch_embeddings = [np.zeros(1536) for _ in batch]  # OpenAI embedding dimension
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _attribute_sentence(self, 
                                sentence: str,
                                sentence_index: int,
                                sentence_embedding: np.ndarray,
                                source_materials: List[Dict[str, Any]]) -> SentenceCitation:
        """Attribute a single sentence to the best supporting source."""
        
        best_match = None
        best_score = 0.0
        best_span = None
        
        # Try primary attribution method (cosine similarity)
        if self.config.primary_attribution_method == "cosine_similarity":
            best_match, best_score, best_span = await self._cosine_similarity_attribution(
                sentence, sentence_embedding, source_materials
            )
        
        # Fallback to keyword overlap if cosine similarity fails
        if best_score < self.config.warning_threshold and self.config.fallback_attribution_method == "keyword_overlap":
            fallback_match, fallback_score, fallback_span = self._keyword_overlap_attribution(
                sentence, source_materials
            )
            if fallback_score > best_score:
                best_match, best_score, best_span = fallback_match, fallback_score, fallback_span
        
        # Determine confidence level and warning status
        confidence_level, needs_warning = self._calculate_confidence(best_score)
        
        # Create citation
        if best_match and best_span:
            citation = SentenceCitation(
                sentence=sentence,
                sentence_index=sentence_index,
                source_document_id=best_match["document_id"],
                source_doc_name=best_match["doc_name"],
                source_chunk_index=best_match["chunk_index"],
                supporting_span=best_span,
                attribution_score=best_score,
                attribution_method=self.config.primary_attribution_method,
                confidence_level=confidence_level,
                needs_warning=needs_warning
            )
        else:
            # Create low-confidence citation with warning
            citation = SentenceCitation(
                sentence=sentence,
                sentence_index=sentence_index,
                source_document_id="unknown",
                source_doc_name="No supporting source found",
                source_chunk_index=None,
                supporting_span=TextSpan(start=0, end=0, text=""),
                attribution_score=0.0,
                attribution_method="none",
                confidence_level="low",
                needs_warning=True
            )
        
        return citation
    
    async def _cosine_similarity_attribution(self, 
                                           sentence: str,
                                           sentence_embedding: np.ndarray,
                                           source_materials: List[Dict[str, Any]]) -> Tuple[Optional[Dict], float, Optional[TextSpan]]:
        """Attribute sentence using cosine similarity between embeddings."""
        
        best_match = None
        best_score = 0.0
        best_span = None
        
        for material in source_materials:
            try:
                # Generate embedding for the source material
                source_embedding = await self.retriever.embed_query(material["text"])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(sentence_embedding, source_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = material
                    
                    # Find the best supporting span within the source text
                    best_span = self._find_best_span(sentence, material["text"], similarity)
                
            except Exception as e:
                logger.warning(f"Cosine similarity failed for material {material['document_id']}: {e}")
                continue
        
        return best_match, best_score, best_span
    
    def _keyword_overlap_attribution(self, 
                                   sentence: str,
                                   source_materials: List[Dict[str, Any]]) -> Tuple[Optional[Dict], float, Optional[TextSpan]]:
        """Fallback attribution using keyword overlap."""
        
        # Extract keywords from sentence (simple approach)
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        sentence_words = {word for word in sentence_words if len(word) > 3}  # Filter short words
        
        best_match = None
        best_score = 0.0
        best_span = None
        
        for material in source_materials:
            # Extract keywords from source material
            source_words = set(re.findall(r'\b\w+\b', material["text"].lower()))
            
            # Calculate Jaccard similarity
            intersection = sentence_words.intersection(source_words)
            union = sentence_words.union(source_words)
            
            if union:
                jaccard_score = len(intersection) / len(union)
                
                if jaccard_score > best_score:
                    best_score = jaccard_score
                    best_match = material
                    
                    # Find span with highest keyword overlap
                    best_span = self._find_keyword_span(sentence, material["text"], intersection)
        
        return best_match, best_score, best_span
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Ensure the result is in [0, 1] range
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _find_best_span(self, sentence: str, source_text: str, similarity_score: float) -> TextSpan:
        """Find the best supporting span within source text."""
        
        # For now, use a simple approach: find the sentence in source text
        # that has the highest word overlap with the target sentence
        source_sentences = sent_tokenize(source_text)
        
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        best_span_start = 0
        best_span_text = source_text[:100]  # Default to first 100 chars
        best_overlap = 0
        
        current_pos = 0
        for source_sentence in source_sentences:
            source_words = set(re.findall(r'\b\w+\b', source_sentence.lower()))
            overlap = len(sentence_words.intersection(source_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_span_start = current_pos
                best_span_text = source_sentence
            
            current_pos = source_text.find(source_sentence, current_pos) + len(source_sentence)
        
        return TextSpan(
            start=best_span_start,
            end=best_span_start + len(best_span_text),
            text=best_span_text.strip()
        )
    
    def _find_keyword_span(self, sentence: str, source_text: str, keywords: set) -> TextSpan:
        """Find span with highest keyword overlap."""
        
        # Simple approach: find the first occurrence of any keyword
        for keyword in keywords:
            pos = source_text.lower().find(keyword)
            if pos != -1:
                # Expand around the keyword to create a meaningful span
                start = max(0, pos - 50)
                end = min(len(source_text), pos + len(keyword) + 50)
                
                return TextSpan(
                    start=start,
                    end=end,
                    text=source_text[start:end].strip()
                )
        
        # Fallback: return first 100 characters
        return TextSpan(
            start=0,
            end=min(100, len(source_text)),
            text=source_text[:100].strip()
        )
    
    def _calculate_confidence(self, score: float) -> Tuple[str, bool]:
        """Calculate confidence level and warning status from attribution score."""
        
        if score >= self.config.high_confidence_threshold:
            return "high", False
        elif score >= self.config.medium_confidence_threshold:
            return "medium", False
        elif score >= self.config.warning_threshold:
            return "low", False
        else:
            return "low", True
    
    def _create_attribution_result(self, 
                                 response_text: str,
                                 sentences: List[str],
                                 sentence_citations: List[SentenceCitation],
                                 attribution_time_ms: float) -> SentenceAttributionResult:
        """Create the final attribution result."""
        
        # Calculate metrics
        sentences_with_citations = sum(1 for c in sentence_citations if c.attribution_score > 0)
        sentences_with_warnings = sum(1 for c in sentence_citations if c.needs_warning)
        attribution_coverage = sentences_with_citations / len(sentences) if sentences else 0.0
        
        # Calculate overall confidence (average of non-zero scores)
        valid_scores = [c.attribution_score for c in sentence_citations if c.attribution_score > 0]
        overall_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Get unique sources and usage counts
        unique_sources = list(set(c.source_document_id for c in sentence_citations if c.source_document_id != "unknown"))
        source_usage_counts = {}
        for citation in sentence_citations:
            if citation.source_document_id != "unknown":
                source_usage_counts[citation.source_document_id] = source_usage_counts.get(citation.source_document_id, 0) + 1
        
        return SentenceAttributionResult(
            response_text=response_text,
            sentences=sentences,
            sentence_citations=sentence_citations,
            overall_confidence=overall_confidence,
            sentences_with_citations=sentences_with_citations,
            sentences_with_warnings=sentences_with_warnings,
            attribution_coverage=attribution_coverage,
            attribution_time_ms=attribution_time_ms,
            unique_sources=unique_sources,
            source_usage_counts=source_usage_counts
        )
    
    def _create_empty_result(self, response_text: str, reason: str) -> SentenceAttributionResult:
        """Create an empty result when attribution fails."""
        logger.warning(f"Creating empty attribution result: {reason}")
        
        return SentenceAttributionResult(
            response_text=response_text,
            sentences=[],
            sentence_citations=[],
            overall_confidence=0.0,
            sentences_with_citations=0,
            sentences_with_warnings=0,
            attribution_coverage=0.0,
            attribution_time_ms=0.0,
            unique_sources=[],
            source_usage_counts={}
        )
    
    def _update_stats(self, num_sentences: int, time_ms: float, confidence: float):
        """Update performance statistics."""
        self.stats["total_attributions"] += 1
        self.stats["total_sentences_processed"] += num_sentences
        self.stats["total_time_ms"] += time_ms
        
        # Update rolling average confidence
        total_attributions = self.stats["total_attributions"]
        current_avg = self.stats["average_confidence"]
        self.stats["average_confidence"] = ((current_avg * (total_attributions - 1)) + confidence) / total_attributions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_time = self.stats["total_time_ms"]
        total_attributions = self.stats["total_attributions"]
        
        return {
            **self.stats,
            "average_time_per_attribution_ms": total_time / total_attributions if total_attributions > 0 else 0.0,
            "average_sentences_per_attribution": self.stats["total_sentences_processed"] / total_attributions if total_attributions > 0 else 0.0
        }


async def create_citation_engine(retriever: HybridRetriever, 
                               llm_client: Optional[LLMClient] = None,
                               config: Optional[CitationEngineConfig] = None) -> SentenceCitationEngine:
    """
    Factory function to create a configured citation engine.
    
    Args:
        retriever: HybridRetriever instance
        llm_client: Optional LLM client for enhanced features
        config: Optional configuration
        
    Returns:
        Configured SentenceCitationEngine instance
    """
    return SentenceCitationEngine(
        retriever=retriever,
        llm_client=llm_client,
        config=config
    )
