"""
Data models for MAI Storage RAG API retrieval system.

This module defines Pydantic models for request/response structures,
document representation, and retrieval operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class DistanceMetric(str, Enum):
    """Distance metrics supported by Qdrant."""
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    
    @classmethod
    def from_qdrant(cls, value: str):
        """Convert Qdrant distance metric to our enum."""
        mapping = {
            "Cosine": cls.COSINE,
            "Dot": cls.DOT,
            "Euclidean": cls.EUCLIDEAN,
            "Manhattan": cls.MANHATTAN
        }
        return mapping.get(value, cls.COSINE)


class RetrievalMethod(str, Enum):
    """Retrieval methods available."""
    DENSE = "dense"  # Vector similarity search
    BM25 = "bm25"   # Lexical/keyword search
    HYBRID = "hybrid"  # Combined dense + BM25
    
    
class RerankMethod(str, Enum):
    """Reranking methods available."""
    NONE = "none"
    BGE_RERANKER_V2 = "bge-reranker-v2"
    CROSS_ENCODER = "cross-encoder"


class Document(BaseModel):
    """Represents a document stored in the vector database."""
    
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Optional fields from ingestion
    doc_name: Optional[str] = Field(None, description="Original document name")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")
    total_chunks: Optional[int] = Field(None, description="Total chunks in document")
    file_type: Optional[str] = Field(None, description="Original file type")
    char_count: Optional[int] = Field(None, description="Character count of chunk")
    timestamp: Optional[datetime] = Field(None, description="Ingestion timestamp")
    start_index: Optional[int] = Field(None, description="Start index in original document")


class RetrievalResult(BaseModel):
    """Represents a single retrieval result with scores."""
    
    document: Document
    scores: Dict[str, float] = Field(default_factory=dict, description="Various relevance scores")
    
    # Core scores
    dense_score: Optional[float] = Field(None, description="Dense vector similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 lexical score")
    hybrid_score: Optional[float] = Field(None, description="Combined hybrid score")
    rerank_score: Optional[float] = Field(None, description="Reranker model score")
    final_score: Optional[float] = Field(None, description="Final ranking score")
    
    # Ranking positions
    dense_rank: Optional[int] = Field(None, description="Rank in dense retrieval")
    bm25_rank: Optional[int] = Field(None, description="Rank in BM25 retrieval")
    hybrid_rank: Optional[int] = Field(None, description="Rank after hybrid fusion")
    final_rank: Optional[int] = Field(None, description="Final rank after all processing")


class RetrievalRequest(BaseModel):
    """Request for document retrieval."""
    
    query: str = Field(..., description="Search query text", min_length=1)
    method: RetrievalMethod = Field(default=RetrievalMethod.HYBRID, description="Retrieval method to use")
    
    # Retrieval parameters
    top_k: int = Field(default=20, description="Number of documents to retrieve initially", ge=1, le=100)
    top_k_dense: int = Field(default=50, description="Number of dense results for fusion", ge=1, le=200)
    top_k_bm25: int = Field(default=50, description="Number of BM25 results for fusion", ge=1, le=200)
    
    # Reranking parameters
    rerank_method: RerankMethod = Field(default=RerankMethod.BGE_RERANKER_V2, description="Reranking method")
    rerank_top_k: int = Field(default=100, description="Number of docs to rerank", ge=1, le=500)
    
    # MMR parameters for diversity
    enable_mmr: bool = Field(default=True, description="Enable MMR for diversity")
    mmr_lambda: float = Field(default=0.5, description="MMR lambda parameter (0=diversity, 1=relevance)", ge=0.0, le=1.0)
    
    # Fusion parameters
    rrf_k: int = Field(default=60, description="RRF k parameter for score fusion", ge=1)
    dense_weight: float = Field(default=0.5, description="Weight for dense scores in fusion", ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, description="Weight for BM25 scores in fusion", ge=0.0, le=1.0)
    
    # Collection parameters
    collection_name: str = Field(default="maistorage_documents", description="Qdrant collection name")
    
    # Filtering
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")
    
    # Output control
    include_metadata: bool = Field(default=True, description="Include document metadata in results")
    include_scores: bool = Field(default=True, description="Include detailed scoring information")


class RetrievalResponse(BaseModel):
    """Response from document retrieval."""
    
    query: str = Field(..., description="Original query")
    results: List[RetrievalResult] = Field(..., description="Retrieved documents with scores")
    
    # Retrieval metadata
    total_results: int = Field(..., description="Total number of results returned")
    method_used: RetrievalMethod = Field(..., description="Retrieval method that was used")
    rerank_method: Optional[RerankMethod] = Field(None, description="Reranking method applied")
    
    # Performance metrics
    retrieval_time_ms: float = Field(..., description="Total retrieval time in milliseconds")
    dense_time_ms: Optional[float] = Field(None, description="Dense retrieval time")
    bm25_time_ms: Optional[float] = Field(None, description="BM25 retrieval time")
    fusion_time_ms: Optional[float] = Field(None, description="Result fusion time")
    rerank_time_ms: Optional[float] = Field(None, description="Reranking time")
    mmr_time_ms: Optional[float] = Field(None, description="MMR processing time")
    
    # Collection info
    collection_name: str = Field(..., description="Collection searched")
    collection_size: Optional[int] = Field(None, description="Total documents in collection")
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information")


class SearchStats(BaseModel):
    """Statistics about search performance and results."""
    
    total_queries: int = Field(default=0, description="Total queries processed")
    avg_retrieval_time_ms: float = Field(default=0.0, description="Average retrieval time")
    avg_results_per_query: float = Field(default=0.0, description="Average results returned")
    
    # Method breakdown
    dense_queries: int = Field(default=0, description="Dense-only queries")
    bm25_queries: int = Field(default=0, description="BM25-only queries")
    hybrid_queries: int = Field(default=0, description="Hybrid queries")
    
    # Performance percentiles
    p50_time_ms: float = Field(default=0.0, description="50th percentile retrieval time")
    p95_time_ms: float = Field(default=0.0, description="95th percentile retrieval time")
    p99_time_ms: float = Field(default=0.0, description="99th percentile retrieval time")


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""
    
    name: str = Field(..., description="Collection name")
    vectors_count: int = Field(..., description="Number of vectors in collection")
    indexed_documents: int = Field(..., description="Number of unique documents")
    vector_size: int = Field(..., description="Vector dimensionality")
    distance_metric: DistanceMetric = Field(..., description="Distance metric used")
    
    # Index configuration
    index_type: str = Field(..., description="Index type (e.g., HNSW)")
    index_params: Dict[str, Any] = Field(default_factory=dict, description="Index parameters")
    
    # Storage stats
    disk_usage_bytes: Optional[int] = Field(None, description="Disk usage in bytes")
    ram_usage_bytes: Optional[int] = Field(None, description="RAM usage in bytes")
    
    # Metadata
    created_at: Optional[datetime] = Field(None, description="Collection creation time")
    last_updated: Optional[datetime] = Field(None, description="Last update time")


class HealthCheck(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Check timestamp (ISO format)")
    version: str = Field(..., description="API version")
    
    # Component health
    qdrant_healthy: bool = Field(..., description="Qdrant connection status")
    embeddings_healthy: bool = Field(..., description="Embeddings service status")
    reranker_healthy: bool = Field(..., description="Reranker model status")
    
    # Performance
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_requests: int = Field(default=0, description="Total requests processed")
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")


# Error models
class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: ErrorDetail = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


# Configuration models
class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""
    
    # Default parameters
    default_top_k: int = Field(default=20, description="Default top-k results")
    default_rerank_top_k: int = Field(default=100, description="Default rerank candidates")
    max_top_k: int = Field(default=100, description="Maximum allowed top-k")
    
    # Model configurations
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker model name")
    
    # Performance limits
    max_query_length: int = Field(default=512, description="Maximum query length in tokens")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")
    
    # Caching
    enable_cache: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # BM25 configuration
    bm25_k1: float = Field(default=1.2, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    
    # HNSW configuration for Qdrant
    hnsw_ef_construct: int = Field(default=256, description="HNSW ef_construct parameter")
    hnsw_m: int = Field(default=16, description="HNSW m parameter")
    hnsw_ef_search: int = Field(default=64, description="HNSW ef search parameter")


# Phase 6: Sentence-level citation models
class TextSpan(BaseModel):
    """Represents a span of text within a larger document."""
    
    start: int = Field(..., description="Starting character index")
    end: int = Field(..., description="Ending character index")
    text: str = Field(..., description="Text content of the span")


class SentenceCitation(BaseModel):
    """Citation information for a single sentence with confidence."""
    
    sentence: str = Field(..., description="The sentence being cited")
    sentence_index: int = Field(..., description="Index of sentence in the response")
    
    # Source information
    source_document_id: str = Field(..., description="ID of supporting document")
    source_doc_name: Optional[str] = Field(None, description="Name of supporting document")
    source_chunk_index: Optional[int] = Field(None, description="Chunk index in document")
    
    # Attribution details
    supporting_span: TextSpan = Field(..., description="Specific text span that supports this sentence")
    attribution_score: float = Field(..., description="Confidence score for this attribution", ge=0.0, le=1.0)
    attribution_method: str = Field(..., description="Method used for attribution (e.g., 'cosine_similarity', 'llm_alignment')")
    
    # Quality indicators
    confidence_level: str = Field(..., description="Confidence level: 'high', 'medium', 'low'")
    needs_warning: bool = Field(default=False, description="Whether to show ⚠️ warning for low confidence")
    
    # Optional enhancement
    rephrased_sentence: Optional[str] = Field(None, description="LLM-enhanced sentence for better alignment")


class SentenceAttributionResult(BaseModel):
    """Complete sentence-level attribution for a response."""
    
    response_text: str = Field(..., description="Full response text")
    sentences: List[str] = Field(..., description="Individual sentences extracted from response")
    
    # Attribution results
    sentence_citations: List[SentenceCitation] = Field(..., description="Citation for each sentence")
    overall_confidence: float = Field(..., description="Overall confidence across all sentences", ge=0.0, le=1.0)
    
    # Quality metrics
    sentences_with_citations: int = Field(..., description="Number of sentences with citations")
    sentences_with_warnings: int = Field(..., description="Number of sentences with low confidence warnings")
    attribution_coverage: float = Field(..., description="Percentage of sentences with citations", ge=0.0, le=1.0)
    
    # Performance
    attribution_time_ms: float = Field(..., description="Time taken for attribution in milliseconds")
    
    # Source mapping
    unique_sources: List[str] = Field(..., description="List of unique source document IDs")
    source_usage_counts: Dict[str, int] = Field(..., description="How many sentences cite each source")


class EnhancedCitation(BaseModel):
    """Enhanced citation combining chunk-level and sentence-level information."""
    
    # Legacy chunk-level fields (for backward compatibility)
    document_id: str = Field(..., description="Document identifier")
    doc_name: Optional[str] = Field(None, description="Document name")
    chunk_index: Optional[int] = Field(None, description="Chunk index")
    score: float = Field(..., description="Chunk relevance score")
    text_snippet: str = Field(..., description="Relevant text snippet from chunk")
    
    # New sentence-level attribution
    sentence_attributions: List[SentenceCitation] = Field(default_factory=list, description="Sentence-level attributions for this source")
    attribution_summary: Optional[str] = Field(None, description="Summary of what this source supports")
    confidence_level: str = Field(default="medium", description="Overall confidence for this source")


class CitationEngineConfig(BaseModel):
    """Configuration for the sentence-level citation engine."""
    
    # Attribution thresholds
    high_confidence_threshold: float = Field(default=0.8, description="Threshold for high confidence", ge=0.0, le=1.0)
    medium_confidence_threshold: float = Field(default=0.6, description="Threshold for medium confidence", ge=0.0, le=1.0)
    warning_threshold: float = Field(default=0.4, description="Below this threshold, show warning", ge=0.0, le=1.0)
    
    # Processing options
    enable_sentence_rephrasing: bool = Field(default=False, description="Enable LLM-based sentence rephrasing")
    min_sentence_length: int = Field(default=10, description="Minimum sentence length to process")
    max_sentences_per_response: int = Field(default=50, description="Maximum sentences to process")
    
    # Attribution method
    primary_attribution_method: str = Field(default="cosine_similarity", description="Primary attribution method")
    fallback_attribution_method: str = Field(default="keyword_overlap", description="Fallback attribution method")
    
    # Performance limits
    max_attribution_time_ms: float = Field(default=5000.0, description="Maximum time to spend on attribution")
    batch_size: int = Field(default=10, description="Batch size for sentence processing")
