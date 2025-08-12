"""
FastAPI main application for MAI Storage RAG API.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, AsyncGenerator
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Ensure asyncio compatibility for RAGAS
import asyncio
import sys

# Force standard asyncio event loop policy to avoid uvloop conflicts
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Import nest_asyncio and patch if needed for RAGAS compatibility  
import nest_asyncio
try:
    # Check if we're using uvloop
    if hasattr(asyncio, '_get_running_loop'):
        current_loop = asyncio._get_running_loop()
        if current_loop and 'uvloop' in str(type(current_loop)):
            print("Warning: uvloop detected, skipping nest_asyncio patching to avoid conflicts")
        else:
            nest_asyncio.apply()
    else:
        # Fallback for older Python versions
        nest_asyncio.apply()
except Exception as e:
    print(f"Note: nest_asyncio patching skipped: {e}")

# Import RAGAS components with better error handling
RAGEvaluator = None
RAGAS_AVAILABLE = False

try:
    from eval.run_ragas import RAGEvaluator
    RAGAS_AVAILABLE = True
    print("RAGAS evaluation system loaded successfully")
except ImportError as e:
    print(f"RAGAS import failed - some dependencies may be missing: {e}")
except Exception as e:
    print(f"RAGAS evaluation unavailable due to initialization error: {e}")
    if "uvloop" in str(e).lower() or "nest_asyncio" in str(e).lower():
        print("This appears to be an event loop compatibility issue. Will attempt to resolve at runtime.")

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import validator, Field
from pydantic_settings import BaseSettings
import uvicorn

# Phase 10 imports
from logging_config import (
    RequestContextMiddleware, 
    setup_development_logging, 
    setup_production_logging,
    get_logger,
    PerformanceLogger
)
from middleware import (
    RequestSizeLimitMiddleware,
    RateLimitMiddleware,
    CircuitBreakerMiddleware,
    SecurityHeadersMiddleware,
    ErrorBoundaryMiddleware,
    DateTimeAwareJSONResponse
)

# Local imports
from models import (
    RetrievalRequest, RetrievalResponse, HealthCheck, 
    ErrorResponse, ErrorDetail, CollectionInfo
)
import psutil
import subprocess
from retrieval import HybridRetriever
from rag_baseline import BaselineRAG, RAGRequest, RAGResponse, create_baseline_rag
from llm_client import LLMConfig
from graph import AgenticRAG, create_agentic_rag, TraceEvent, TraceEventType
from monitoring import (
    get_metrics_collector,
    metrics_endpoint,
    stats_endpoint,
    health_detailed_endpoint
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with comprehensive environment validation."""
    
    # Core application settings
    app_name: str = "MAI Storage RAG API"
    environment: str = Field(default="development", description="Deployment environment")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # API Keys and authentication
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    
    # Model configuration
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for generation")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker model")
    
    # Service URLs
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant database URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (optional)")
    elasticsearch_url: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    
    # Performance and limits
    max_request_size: int = Field(default=10_000_000, description="Max request size in bytes (10MB)")
    max_query_length: int = Field(default=8192, description="Max query length in characters")
    max_top_k: int = Field(default=100, description="Maximum top_k for retrieval")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_burst: int = Field(default=20, description="Rate limit burst capacity")
    
    # Tracing and monitoring
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangChain tracing")
    
    # Database connection settings
    qdrant_timeout: int = Field(default=60, description="Qdrant connection timeout")
    qdrant_prefer_grpc: bool = Field(default=False, description="Prefer gRPC for Qdrant")
    
    # Cache settings
    enable_cache: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    @validator('openai_api_key')
    def validate_openai_api_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError('OPENAI_API_KEY must be provided and start with "sk-"')
        return v
    
    @validator('qdrant_url')
    def validate_qdrant_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('QDRANT_URL must be a valid HTTP/HTTPS URL')
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f'ENVIRONMENT must be one of: {allowed_envs}')
        return v
    
    @validator('max_request_size')
    def validate_max_request_size(cls, v):
        if v < 1_000_000 or v > 100_000_000:  # 1MB to 100MB
            raise ValueError('MAX_REQUEST_SIZE must be between 1MB and 100MB')
        return v
    
    @validator('max_query_length')
    def validate_max_query_length(cls, v):
        if v < 10 or v > 32768:  # 10 chars to 32k chars
            raise ValueError('MAX_QUERY_LENGTH must be between 10 and 32768 characters')
        return v
    
    @validator('rate_limit_requests')
    def validate_rate_limit_requests(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('RATE_LIMIT_REQUESTS must be between 1 and 10000')
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def get_log_config(self) -> dict:
        """Get logging configuration based on environment."""
        return {
            "level": self.log_level,
            "json_logs": self.is_production(),
            "enable_tracing": self.enable_tracing
        }
    
    class Config:
        env_file = "../../.env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        case_sensitive = False


settings = Settings()

# Initialize logging based on environment
if settings.is_production():
    setup_production_logging()
else:
    setup_development_logging()

# Get structured logger
logger = get_logger(__name__)

# Initialize tracing if enabled
from logging_config import get_tracing_adapter
tracing_adapter = get_tracing_adapter()
if settings.enable_tracing:
    logger.info("tracing_enabled", 
                langfuse=bool(settings.langfuse_public_key),
                langchain=settings.langchain_tracing_v2)

app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG API with hybrid retrieval, reranking, and MMR diversity",
    version="0.3.0",  # Phase 10 version
)

# Add middleware stack (order matters - last added runs first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ErrorBoundaryMiddleware)
app.add_middleware(CircuitBreakerMiddleware, 
                   failure_threshold=5, 
                   recovery_timeout=60)
app.add_middleware(RateLimitMiddleware,
                   requests_per_minute=settings.rate_limit_requests,
                   burst_capacity=settings.rate_limit_burst)
app.add_middleware(RequestSizeLimitMiddleware, 
                   max_request_size=settings.max_request_size)
app.add_middleware(RequestContextMiddleware)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",  # Allow all Vercel subdomains
        "https://vercel.app",    # Allow Vercel domains
        "https://your-actual-vercel-url.vercel.app",  # Add your specific URL here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
retriever: Optional[HybridRetriever] = None
baseline_rag: Optional[BaselineRAG] = None
agentic_rag: Optional[AgenticRAG] = None
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global retriever, baseline_rag, agentic_rag
    
    logger.info("Starting MAI Storage RAG API...")
    
    try:
        # Initialize hybrid retriever
        retriever = HybridRetriever(
            qdrant_url=settings.qdrant_url,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
            openai_api_key=settings.openai_api_key
        )
        
        logger.info("Hybrid retriever initialized successfully")
        
        # Initialize baseline RAG system
        baseline_rag = create_baseline_rag(
            retriever=retriever,
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
            enable_sentence_citations=False  # Can be configured later
        )
        
        logger.info("Baseline RAG system initialized successfully")
        
        # Initialize agentic RAG system (Phase 5)
        agentic_rag = create_agentic_rag(
            retriever=retriever,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7,
            enable_sentence_citations=False  # Can be configured later
        )
        
        logger.info("Agentic RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MAI Storage RAG API...")


def get_retriever() -> HybridRetriever:
    """Dependency to get the retriever instance."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval service not initialized"
        )
    return retriever


def get_baseline_rag() -> BaselineRAG:
    """Dependency to get the baseline RAG instance."""
    if baseline_rag is None:
        raise HTTPException(
            status_code=503,
            detail="Baseline RAG service not initialized"
        )
    return baseline_rag


def get_agentic_rag() -> AgenticRAG:
    """Dependency to get the agentic RAG instance."""
    if agentic_rag is None:
        raise HTTPException(
            status_code=503,
            detail="Agentic RAG service not initialized"
        )
    return agentic_rag


@app.get("/", response_model=dict)
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "MAI Storage RAG API is running",
        "app_name": settings.app_name,
        "version": "0.3.0",
        "model": settings.openai_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "features": {
            "hybrid_retrieval": "Dense vector + BM25 + RRF fusion",
            "reranking": "Cross-encoder reranking with BGE-reranker-v2",
            "baseline_rag": "Traditional RAG with citations",
            "agentic_rag": "Multi-step agentic RAG with LangGraph (Phase 5)",
            "endpoints": {
                "retrieve": "POST /retrieve - Hybrid document retrieval",
                "rag": "POST /rag - Baseline RAG generation with citations",
                "chat/stream": "POST /chat/stream - Streaming agentic/baseline RAG with trace events",
                "health": "GET /health - System health check",
                "stats": "GET /stats - Performance metrics"
            }
        },
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check(retriever_instance: HybridRetriever = Depends(get_retriever)):
    """Comprehensive health check with enhanced diagnostics."""
    uptime_seconds = time.time() - app_start_time
    
    # Check component health
    qdrant_healthy = True
    embeddings_healthy = True
    reranker_healthy = True
    
    try:
        # Test Qdrant connection with timeout
        collections = retriever_instance.qdrant_client.get_collections()
        logger.debug(f"Qdrant collections: {len(collections.collections) if collections else 0}")
    except Exception as e:
        logger.warning("qdrant_health_check_failed", error=str(e))
        qdrant_healthy = False
    
    try:
        # Test embeddings (quick test with timeout)
        with PerformanceLogger("health_check_embedding"):
            await retriever_instance.embed_query("health check")
    except Exception as e:
        logger.warning("embeddings_health_check_failed", error=str(e))
        embeddings_healthy = False
    
    try:
        # Test reranker model (lazy loading check)
        _ = retriever_instance.reranker_model
    except Exception as e:
        logger.warning("reranker_health_check_failed", error=str(e))
        reranker_healthy = False
    
    # Get system metrics
    try:
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_usage_mb = memory_info.used / 1024 / 1024
    except Exception as e:
        logger.warning("system_metrics_failed", error=str(e))
        memory_usage_mb = None
        cpu_percent = None
    
    # Get retrieval stats (synchronous access to avoid async issues)
    stats = retriever_instance.stats.copy()
    
    # Determine overall status
    status = "healthy" if all([qdrant_healthy, embeddings_healthy, reranker_healthy]) else "unhealthy"
    
    return HealthCheck(
        status=status,
        timestamp=datetime.utcnow().isoformat(),
        version="0.3.0",  # Updated version
        qdrant_healthy=qdrant_healthy,
        embeddings_healthy=embeddings_healthy,
        reranker_healthy=reranker_healthy,
        uptime_seconds=uptime_seconds,
        total_requests=stats.get("total_queries", 0),
        memory_usage_mb=memory_usage_mb,
        cpu_usage_percent=cpu_percent
    )


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
    retriever_instance: HybridRetriever = Depends(get_retriever)
):
    """
    Retrieve documents using hybrid search with optional reranking and MMR.
    
    This endpoint implements the complete Phase 2 retrieval pipeline:
    - Dense vector search using OpenAI embeddings
    - BM25 lexical search for keyword matching
    - Reciprocal Rank Fusion (RRF) for combining results
    - Cross-encoder reranking using bge-reranker-v2
    - Maximal Marginal Relevance (MMR) for diversity
    
    Args:
        request: Retrieval request with query and parameters
        
    Returns:
        Retrieval response with ranked documents and metadata
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Retrieval request: method={request.method}, query_length={len(request.query)}")
        
        # Perform retrieval
        response = await retriever_instance.retrieve(request)
        
        logger.info(f"Retrieval completed: {response.total_results} results in {response.retrieval_time_ms:.2f}ms")
        return response
        
    except ValueError as e:
        # Client error (bad request)
        logger.warning(f"Invalid retrieval request: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_REQUEST",
                    message=str(e)
                )
            ).dict()
        )
    
    except Exception as e:
        # Server error
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="RETRIEVAL_FAILED",
                    message="Internal server error during retrieval"
                )
            ).dict()
        )


@app.get("/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection_info(
    collection_name: str,
    retriever_instance: HybridRetriever = Depends(get_retriever)
):
    """Get information about a specific collection."""
    try:
        collection_info = await retriever_instance.get_collection_info(collection_name)
        
        if collection_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        return collection_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve collection information"
        )


@app.post("/rag", response_model=RAGResponse)
async def generate_rag_answer(
    request: RAGRequest,
    rag_instance: BaselineRAG = Depends(get_baseline_rag)
):
    """
    Generate answer using baseline RAG approach.
    
    This endpoint implements the Phase 3 baseline RAG functionality:
    - Query → retrieve top-k chunks using hybrid search
    - Pack context → LLM generate answer using gpt-4o-mini
    - Return answer with chunk-level citations
    
    Args:
        request: RAG request with query and parameters
        
    Returns:
        RAG response with generated answer and source citations
        
    Raises:
        HTTPException: If RAG generation fails
    """
    try:
        logger.info(f"RAG request: query_length={len(request.query)}, top_k={request.top_k}")
        
        # Generate RAG response
        response = await rag_instance.generate(request)
        
        logger.info(f"RAG completed: {len(response.citations)} citations, {response.total_time_ms:.2f}ms total")
        return response
        
    except ValueError as e:
        # Client error (bad request)
        logger.warning(f"Invalid RAG request: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_REQUEST",
                    message=str(e)
                )
            ).dict()
        )
    
    except Exception as e:
        # Server error
        logger.error(f"RAG generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="RAG_GENERATION_FAILED",
                    message="Internal server error during RAG generation"
                )
            ).dict()
        )


# Phase 10 monitoring endpoints
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return await metrics_endpoint()


@app.get("/stats", response_model=dict)
async def get_comprehensive_stats():
    """Get comprehensive system and performance statistics."""
    return await stats_endpoint()


@app.get("/health/detailed", response_model=dict)
async def get_detailed_health():
    """Get detailed health information including dependencies."""
    return await health_detailed_endpoint()


@app.get("/stats/legacy", response_model=dict)
async def get_stats_legacy(retriever_instance: HybridRetriever = Depends(get_retriever)):
    """Get retrieval statistics and performance metrics."""
    try:
        stats = retriever_instance.get_stats()
        
        # Add additional API stats
        stats.update({
            "uptime_seconds": time.time() - app_start_time,
            "app_version": "0.2.0",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve statistics"
        )


@app.post("/cache/clear", response_model=dict)
async def clear_cache(retriever_instance: HybridRetriever = Depends(get_retriever)):
    """Clear the BM25 cache."""
    try:
        retriever_instance.clear_cache()
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear cache"
        )


# Pydantic models for agentic chat streaming
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    """Request for agentic chat streaming."""
    
    query: str = Field(..., description="User query", min_length=1)
    top_k: int = Field(default=10, description="Number of documents to retrieve", ge=1, le=50)
    enable_verification: bool = Field(default=True, description="Enable answer verification")
    max_refinements: int = Field(default=2, description="Maximum refinement iterations", ge=0, le=5)
    
    # Stream control
    stream_traces: bool = Field(default=True, description="Stream trace events")
    stream_tokens: bool = Field(default=False, description="Stream individual tokens (future feature)")


@app.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    agentic: bool = False,
    agentic_rag_instance: AgenticRAG = Depends(get_agentic_rag),
    baseline_rag_instance: BaselineRAG = Depends(get_baseline_rag)
):
    """
    Stream chat responses with agentic or baseline RAG.
    
    This endpoint implements Phase 5 agentic streaming with NDJSON events:
    - ?agentic=1: Use multi-step agentic RAG with trace events
    - ?agentic=0 (default): Use baseline RAG for comparison
    
    Events emitted:
    - step_start: When a workflow step begins
    - step_complete: When a workflow step completes  
    - sources: Retrieved source documents
    - verification: Verification results
    - metrics: Performance metrics
    - done: Workflow completion
    
    Args:
        request: Chat request with query and parameters
        agentic: Whether to use agentic (True) or baseline (False) RAG
        
    Returns:
        Streaming NDJSON response with trace events and final answer
    """
    try:
        logger.info(f"Chat stream request: agentic={agentic}, query_length={len(request.query)}")
        
        if agentic:
            # Use agentic RAG with streaming trace events
            async def agentic_stream_generator() -> AsyncGenerator[str, None]:
                try:
                    # Run the agentic workflow
                    final_state = await agentic_rag_instance.run(
                        query=request.query,
                        top_k=request.top_k,
                        enable_verification=request.enable_verification,
                        max_refinements=request.max_refinements
                    )
                    
                    # Stream trace events
                    if request.stream_traces:
                        for trace_event in final_state.get("trace_events", []):
                            event_data = {
                                "type": trace_event.event_type.value,
                                "timestamp": trace_event.timestamp.isoformat(),
                                "step": trace_event.step.value if trace_event.step else None,
                                "data": trace_event.data
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            
                            # Small delay to allow client processing
                            await asyncio.sleep(0.01)
                    
                    # Stream final answer
                    answer_data = {
                        "type": "answer",
                        "content": final_state.get("answer", ""),
                        "citations": final_state.get("citations", []),
                        "metadata": {
                            "total_time_ms": final_state.get("total_time_ms", 0),
                            "refinement_count": final_state.get("refinement_count", 0),
                            "step_times": final_state.get("step_times", {}),
                            "verification_passed": not final_state.get("needs_refinement", False),
                            "tokens_used": final_state.get("tokens_used", {}),
                            "total_tokens": final_state.get("tokens_used", {}).get("total_tokens", 0)
                        }
                    }
                    yield f"data: {json.dumps(answer_data)}\n\n"
                    
                    # Final done event
                    done_data = {
                        "type": "done",
                        "success": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
                    
                except Exception as e:
                    logger.error(f"Agentic stream failed: {e}")
                    error_data = {
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                agentic_stream_generator(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-RAG-Type": "agentic"
                }
            )
        
        else:
            # Use baseline RAG with simulated streaming for comparison
            async def baseline_stream_generator() -> AsyncGenerator[str, None]:
                try:
                    # Create RAG request
                    rag_request = RAGRequest(
                        query=request.query,
                        top_k=request.top_k
                    )
                    
                    # Generate baseline response
                    start_time = time.time()
                    rag_response = await baseline_rag_instance.generate(rag_request)
                    total_time = (time.time() - start_time) * 1000
                    
                    # Emit start event
                    start_data = {
                        "type": "step_start",
                        "step": "baseline_rag",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"query": request.query}
                    }
                    yield f"data: {json.dumps(start_data)}\n\n"
                    await asyncio.sleep(0.05)
                    
                    # Emit sources event
                    sources_data = {
                        "type": "sources",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "sources": [
                                {
                                    "doc_name": citation.doc_name,
                                    "chunk_index": citation.chunk_index,
                                    "text_snippet": citation.text_snippet,
                                    "relevance_score": citation.score
                                }
                                for citation in rag_response.citations
                            ]
                        }
                    }
                    yield f"data: {json.dumps(sources_data)}\n\n"
                    await asyncio.sleep(0.05)
                    
                    # Simulate token streaming by splitting answer into words
                    words = rag_response.answer.split()
                    answer_so_far = ""
                    
                    for word in words:
                        answer_so_far += word + " "
                        token_data = {
                            "type": "token",
                            "content": word + " ",
                            "partial_answer": answer_so_far.strip(),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        yield f"data: {json.dumps(token_data)}\n\n"
                        await asyncio.sleep(0.05)  # 50ms delay between words
                    
                    # Emit completion event
                    complete_data = {
                        "type": "step_complete",
                        "step": "baseline_rag",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"time_ms": total_time}
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"
                    
                    # Stream final answer data
                    answer_data = {
                        "type": "answer",
                        "content": rag_response.answer,
                        "citations": [
                            {
                                "doc_name": citation.doc_name,
                                "chunk_index": citation.chunk_index,
                                "text_snippet": citation.text_snippet,
                                "relevance_score": citation.score
                            }
                            for citation in rag_response.citations
                        ],
                        "metadata": {
                            "total_time_ms": total_time,
                            "generation_time_ms": rag_response.generation_time_ms,
                            "retrieval_time_ms": rag_response.retrieval_time_ms,
                            "tokens_used": rag_response.tokens_used
                        }
                    }
                    yield f"data: {json.dumps(answer_data)}\n\n"
                    
                    # Final done event
                    done_data = {
                        "type": "done",
                        "success": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
                    
                except Exception as e:
                    logger.error(f"Baseline stream failed: {e}")
                    error_data = {
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                baseline_stream_generator(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-RAG-Type": "baseline"
                }
            )
    
    except ValueError as e:
        # Client error (bad request)
        logger.warning(f"Invalid chat stream request: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_REQUEST",
                    message=str(e)
                )
            ).dict()
        )
    
    except Exception as e:
        # Server error
        logger.error(f"Chat stream failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="CHAT_STREAM_FAILED",
                    message="Internal server error during chat streaming"
                )
            ).dict()
        )


# ============================================================================
# Phase 8: Evaluation Endpoints
# ============================================================================

@app.post("/eval/run")
async def run_evaluation(request: dict):
    """
    Run evaluation on Traditional or Agentic RAG using golden QA dataset.
    
    This endpoint implements Phase 8 evaluation harness with RAGAS metrics
    and retrieval-specific metrics (Recall@k, nDCG, MRR).
    
    Args:
        request: JSON request body with mode, top_k, save_results
        
    Returns:
        Evaluation results with RAGAS metrics and performance data
    """
    try:
        import json
        
        # Check RAGAS availability with better error reporting
        if not RAGAS_AVAILABLE or RAGEvaluator is None:
            error_msg = "RAGAS evaluation is not available. This could be due to missing dependencies or event loop conflicts."
            logger.error(error_msg)
            return DateTimeAwareJSONResponse(
                status_code=503,
                content={"error": error_msg, "ragas_available": False}
            )
        
        # Extract parameters from request
        mode = request.get("mode", "traditional")
        top_k = request.get("top_k", 5)
        save_results = request.get("save_results", True)
        
        logger.info(f"Starting evaluation - mode={mode}, top_k={top_k}")
        
        # Load golden QA dataset
        golden_qa_path = Path(__file__).parent / "golden_qa.json"
        if not golden_qa_path.exists():
            error_msg = f"Golden QA dataset not found at {golden_qa_path}"
            logger.error(error_msg)
            return DateTimeAwareJSONResponse(
                status_code=404,
                content={"error": error_msg}
            )
            
        with open(golden_qa_path, 'r') as f:
            golden_qa_data = json.load(f)
        
        questions = golden_qa_data["questions"]
        
        # Use global RAG systems (initialized at startup)
        global retriever, baseline_rag, agentic_rag
        
        if retriever is None or baseline_rag is None or agentic_rag is None:
            error_msg = "RAG services not initialized properly"
            logger.error(error_msg)
            return DateTimeAwareJSONResponse(
                status_code=503,
                content={"error": error_msg}
            )
        
        # Try to create evaluator in a safe way that handles event loop issues
        try:
            evaluator = RAGEvaluator(
                baseline_rag=baseline_rag,
                agentic_rag=agentic_rag,
                retriever=retriever
            )
        except Exception as eval_error:
            logger.error(f"Failed to create RAG evaluator: {eval_error}")
            if "uvloop" in str(eval_error).lower() or "nest_asyncio" in str(eval_error).lower():
                error_msg = "Event loop compatibility issue detected. RAGAS requires standard asyncio loop."
                return DateTimeAwareJSONResponse(
                    status_code=503,
                    content={"error": error_msg, "details": str(eval_error)}
                )
            raise
        
        results = {}
        
        # Run Traditional RAG evaluation
        if mode in ["traditional", "both"]:
            logger.info("Running Traditional RAG evaluation...")
            try:
                traditional_results = await evaluator.evaluate_traditional_rag(questions, top_k)
                traditional_results = await evaluator.run_ragas_evaluation(traditional_results)
                traditional_retrieval_metrics = evaluator.calculate_retrieval_metrics(
                    traditional_results, questions, top_k
                )
                
                results["traditional"] = {
                    "results": [result.__dict__ for result in traditional_results],
                    "retrieval_metrics": traditional_retrieval_metrics,
                    "ragas_summary": evaluator._calculate_ragas_summary(traditional_results),
                    "performance_summary": evaluator._calculate_performance_summary(traditional_results)
                }
                
                if save_results:
                    output_dir = Path(__file__).parent / "eval" / "results"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    evaluator.save_results(traditional_results, traditional_retrieval_metrics, output_dir)
            except Exception as e:
                logger.error(f"Traditional RAG evaluation failed: {e}")
                results["traditional"] = {"error": str(e)}
        
        # Run Agentic RAG evaluation  
        if mode in ["agentic", "both"]:
            logger.info("Running Agentic RAG evaluation...")
            try:
                agentic_results = await evaluator.evaluate_agentic_rag(questions, top_k)
                agentic_results = await evaluator.run_ragas_evaluation(agentic_results)
                agentic_retrieval_metrics = evaluator.calculate_retrieval_metrics(
                    agentic_results, questions, top_k
                )
                
                results["agentic"] = {
                    "results": [result.__dict__ for result in agentic_results],
                    "retrieval_metrics": agentic_retrieval_metrics,
                    "ragas_summary": evaluator._calculate_ragas_summary(agentic_results),
                    "performance_summary": evaluator._calculate_performance_summary(agentic_results)
                }
                
                if save_results:
                    output_dir = Path(__file__).parent / "eval" / "results"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    evaluator.save_results(agentic_results, agentic_retrieval_metrics, output_dir)
            except Exception as e:
                logger.error(f"Agentic RAG evaluation failed: {e}")
                results["agentic"] = {"error": str(e)}
        
        # Add metadata
        results["metadata"] = {
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "top_k": top_k,
            "total_questions": len(questions),
            "golden_qa_version": golden_qa_data.get("dataset_info", {}).get("version", "unknown"),
            "ragas_available": RAGAS_AVAILABLE
        }
        
        logger.info(f"Evaluation completed successfully - mode={mode}")
        return DateTimeAwareJSONResponse(content=results)
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Evaluation failed: {e}")
        logger.error(f"Full traceback: {tb_str}")
        
        # Return a proper error response
        return DateTimeAwareJSONResponse(
            status_code=500,
            content={
                "error": f"Evaluation failed: {str(e)}",
                "details": tb_str,
                "ragas_available": RAGAS_AVAILABLE
            }
        )


@app.get("/eval/results")
async def get_evaluation_results(limit: int = 10):
    """
    Get recent evaluation results from saved files.
    
    Returns the most recent evaluation results for display in the frontend.
    
    Args:
        limit: Maximum number of result files to return
        
    Returns:
        List of recent evaluation results with metadata
    """
    try:
        import json
        from pathlib import Path
        import glob
        
        results_dir = Path(__file__).parent / "eval" / "results"
        
        if not results_dir.exists():
            return DateTimeAwareJSONResponse(content={
                "results": [], 
                "message": "No evaluation results found"
            })
        
        # Find all evaluation result JSON files
        json_files = list(results_dir.glob("evaluation_results_*.json"))
        
        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Limit results
        json_files = json_files[:limit]
        
        results = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                    
                # Add file metadata
                result_data["file_info"] = {
                    "filename": file_path.name,
                    "created_time": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "size_bytes": file_path.stat().st_size
                }
                
                results.append(result_data)
                
            except Exception as e:
                logger.warning(f"Failed to load evaluation result {file_path}: {e}")
                continue
        
        return DateTimeAwareJSONResponse(content={
            "results": results,
            "total_files": len(results),
            "results_directory": str(results_dir)
        })
        
    except Exception as e:
        logger.error(f"Failed to get evaluation results: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="RESULTS_RETRIEVAL_FAILED",
                    message="Failed to retrieve evaluation results"
                )
            ).dict()
        )


@app.get("/eval/compare")
async def compare_evaluations(
    traditional_file: Optional[str] = None,
    agentic_file: Optional[str] = None
):
    """
    Compare Traditional vs Agentic RAG evaluation results.
    
    Args:
        traditional_file: Filename of traditional evaluation results
        agentic_file: Filename of agentic evaluation results
        
    Returns:
        Comparison analysis between the two evaluation modes
    """
    try:
        import json
        from pathlib import Path
        
        results_dir = Path(__file__).parent / "eval" / "results"
        
        # If no specific files provided, get the most recent of each type
        if not traditional_file or not agentic_file:
            json_files = list(results_dir.glob("evaluation_results_*.json"))
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            traditional_data = None
            agentic_data = None
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        mode = data.get("metadata", {}).get("mode", "")
                        
                        if "traditional" in mode and not traditional_data:
                            traditional_data = data
                        elif "agentic" in mode and not agentic_data:
                            agentic_data = data
                            
                        if traditional_data and agentic_data:
                            break
                except:
                    continue
        else:
            # Load specific files
            traditional_path = results_dir / traditional_file
            agentic_path = results_dir / agentic_file
            
            with open(traditional_path, 'r') as f:
                traditional_data = json.load(f)
            with open(agentic_path, 'r') as f:
                agentic_data = json.load(f)
        
        if not traditional_data or not agentic_data:
            return DateTimeAwareJSONResponse(content={
                "error": "Could not find both traditional and agentic evaluation results",
                "available_files": [f.name for f in results_dir.glob("*.json")]
            })
        
        # Perform comparison analysis
        comparison = {
            "metadata": {
                "comparison_timestamp": datetime.utcnow().isoformat(),
                "traditional_timestamp": traditional_data.get("metadata", {}).get("evaluation_timestamp"),
                "agentic_timestamp": agentic_data.get("metadata", {}).get("evaluation_timestamp")
            },
            "ragas_comparison": {},
            "retrieval_comparison": {},
            "performance_comparison": {},
            "success_rate_comparison": {}
        }
        
        # Compare RAGAS metrics
        trad_ragas = traditional_data.get("ragas_summary", {})
        agent_ragas = agentic_data.get("ragas_summary", {})
        
        for metric in ["faithfulness_score", "answer_relevancy_score", "context_precision_score", "context_recall_score"]:
            avg_metric = f"avg_{metric}"
            if avg_metric in trad_ragas and avg_metric in agent_ragas:
                comparison["ragas_comparison"][metric] = {
                    "traditional": trad_ragas[avg_metric],
                    "agentic": agent_ragas[avg_metric],
                    "improvement": agent_ragas[avg_metric] - trad_ragas[avg_metric],
                    "improvement_pct": ((agent_ragas[avg_metric] - trad_ragas[avg_metric]) / trad_ragas[avg_metric] * 100) if trad_ragas[avg_metric] > 0 else 0
                }
        
        # Compare retrieval metrics
        trad_retrieval = traditional_data.get("retrieval_metrics", {})
        agent_retrieval = agentic_data.get("retrieval_metrics", {})
        
        for metric in ["recall_at_k", "ndcg_at_k", "mrr_score", "precision_at_k"]:
            if metric in trad_retrieval and metric in agent_retrieval:
                comparison["retrieval_comparison"][metric] = {
                    "traditional": trad_retrieval[metric],
                    "agentic": agent_retrieval[metric], 
                    "improvement": agent_retrieval[metric] - trad_retrieval[metric],
                    "improvement_pct": ((agent_retrieval[metric] - trad_retrieval[metric]) / trad_retrieval[metric] * 100) if trad_retrieval[metric] > 0 else 0
                }
        
        # Compare performance metrics
        trad_perf = traditional_data.get("performance_summary", {})
        agent_perf = agentic_data.get("performance_summary", {})
        
        for metric in ["avg_response_time_ms", "avg_token_usage", "success_rate"]:
            if metric in trad_perf and metric in agent_perf:
                comparison["performance_comparison"][metric] = {
                    "traditional": trad_perf[metric],
                    "agentic": agent_perf[metric],
                    "improvement": agent_perf[metric] - trad_perf[metric],
                    "improvement_pct": ((agent_perf[metric] - trad_perf[metric]) / trad_perf[metric] * 100) if trad_perf[metric] > 0 else 0
                }
        
        return DateTimeAwareJSONResponse(content=comparison)
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="COMPARISON_FAILED",
                    message=f"Failed to compare evaluations: {str(e)}"
                )
            ).dict()
        )


# Phase 9: Performance & Cost Tuning Endpoints

@app.get("/performance/stats")
async def get_performance_stats(retriever_instance: HybridRetriever = Depends(get_retriever)):
    """
    Get comprehensive performance and caching statistics.
    
    Returns detailed metrics including:
    - Retrieval performance (latency, throughput)
    - Cache hit rates and efficiency
    - HNSW configuration status
    - Resource utilization
    """
    try:
        stats = await retriever_instance.get_stats()
        
        # Add system-level performance metrics
        stats["system_metrics"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - app_start_time if 'app_start_time' in globals() else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="STATS_FAILED",
                    message="Failed to retrieve performance statistics"
                )
            ).dict()
        )


@app.post("/cache/clear")
async def clear_caches(cache_type: Optional[str] = None, 
                      retriever_instance: HybridRetriever = Depends(get_retriever)):
    """
    Clear specified cache or all caches.
    
    Args:
        cache_type: Type of cache to clear ('embedding', 'candidate', 'rerank', 'prompt', 'bm25', 'all')
    """
    try:
        if cache_type is None or cache_type == "all":
            # Clear all caches
            if retriever_instance.cache_manager:
                await retriever_instance.cache_manager.clear_all_caches()
            retriever_instance.clear_cache()  # Legacy BM25 cache
            logger.info("All caches cleared")
            return {"message": "All caches cleared successfully"}
        
        # Clear specific cache type
        if cache_type == "embedding" and retriever_instance.cache_manager and retriever_instance.cache_manager.embedding_cache:
            await retriever_instance.cache_manager.embedding_cache.cache.clear()
        elif cache_type == "candidate" and retriever_instance.cache_manager and retriever_instance.cache_manager.candidate_cache:
            await retriever_instance.cache_manager.candidate_cache.cache.clear()
        elif cache_type == "rerank" and retriever_instance.cache_manager and retriever_instance.cache_manager.rerank_cache:
            await retriever_instance.cache_manager.rerank_cache.cache.clear()
        elif cache_type == "prompt" and retriever_instance.cache_manager and retriever_instance.cache_manager.prompt_cache:
            await retriever_instance.cache_manager.prompt_cache.cache.clear()
        elif cache_type == "bm25":
            retriever_instance.clear_cache()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown cache type: {cache_type}. Valid types: embedding, candidate, rerank, prompt, bm25, all"
            )
        
        logger.info(f"Cache type '{cache_type}' cleared")
        return {"message": f"Cache type '{cache_type}' cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="CACHE_CLEAR_FAILED",
                    message=f"Failed to clear cache: {str(e)}"
                )
            ).dict()
        )


@app.get("/tuning/config")
async def get_tuning_config():
    """
    Get current performance tuning configuration.
    
    Returns the current retrieval_tuning.yaml configuration.
    """
    try:
        config_path = "retrieval_tuning.yaml"
        if not os.path.exists(config_path):
            return {"message": "No tuning configuration found", "config": {}}
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return {
            "config_path": config_path,
            "config": config,
            "last_modified": datetime.fromtimestamp(os.path.getmtime(config_path)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get tuning config: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="CONFIG_FAILED",
                    message="Failed to retrieve tuning configuration"
                )
            ).dict()
        )


@app.post("/tuning/benchmark")
async def run_performance_benchmark(
    num_queries: int = 10,
    retriever_instance: HybridRetriever = Depends(get_retriever)
):
    """
    Run a performance benchmark to test current configuration.
    
    Args:
        num_queries: Number of test queries to run
    """
    try:
        import random
        
        # Sample queries for benchmarking
        sample_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "What are the benefits of cloud computing?",
            "How to implement data security?",
            "What is the difference between AI and ML?",
            "Explain deep learning algorithms",
            "How to optimize database performance?",
            "What are microservices architecture patterns?",
            "How to implement REST APIs?"
        ]
        
        # Run benchmark
        start_time = time.time()
        latencies = []
        cache_hits = 0
        total_queries_before = retriever_instance.stats.get("total_queries", 0)
        cache_hits_before = retriever_instance.stats.get("cache_hits", 0)
        
        for i in range(num_queries):
            query = random.choice(sample_queries)
            
            query_start = time.time()
            try:
                # Run a test retrieval
                request = RetrievalRequest(
                    query=query,
                    top_k=5,
                    method="hybrid"
                )
                await retriever_instance.retrieve(request)
                query_latency = (time.time() - query_start) * 1000
                latencies.append(query_latency)
            except Exception as e:
                logger.warning(f"Benchmark query {i+1} failed: {e}")
                continue
        
        total_time = time.time() - start_time
        total_queries_after = retriever_instance.stats.get("total_queries", 0)
        cache_hits_after = retriever_instance.stats.get("cache_hits", 0)
        
        # Calculate statistics
        if latencies:
            import numpy as np
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            mean_latency = np.mean(latencies)
        else:
            p50 = p95 = p99 = mean_latency = 0
        
        benchmark_results = {
            "benchmark_timestamp": datetime.utcnow().isoformat(),
            "configuration": {
                "num_queries": num_queries,
                "queries_completed": len(latencies)
            },
            "latency_metrics": {
                "mean_ms": round(mean_latency, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "min_ms": round(min(latencies), 2) if latencies else 0,
                "max_ms": round(max(latencies), 2) if latencies else 0
            },
            "throughput_metrics": {
                "total_time_s": round(total_time, 2),
                "queries_per_second": round(len(latencies) / total_time, 2) if total_time > 0 else 0
            },
            "cache_metrics": {
                "cache_hits_during_benchmark": cache_hits_after - cache_hits_before,
                "cache_hit_rate": round((cache_hits_after - cache_hits_before) / max(1, total_queries_after - total_queries_before), 3)
            },
            "performance_targets": {
                "p50_target_ms": 800,
                "p95_target_ms": 1500,
                "p50_met": p50 <= 800,
                "p95_met": p95 <= 1500
            }
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="BENCHMARK_FAILED",
                    message=f"Performance benchmark failed: {str(e)}"
                )
            ).dict()
        )


# Track application start time for uptime calculation
app_start_time = time.time()


if __name__ == "__main__":
    import os
    # Use standard asyncio loop to avoid conflicts with RAGAS nest_asyncio
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        loop="asyncio"  # Force standard asyncio instead of uvloop
    )
