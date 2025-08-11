"""
FastAPI main application for MAI Storage RAG API.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
import uvicorn

# Local imports
from models import (
    RetrievalRequest, RetrievalResponse, HealthCheck, 
    ErrorResponse, ErrorDetail, CollectionInfo
)
from retrieval import HybridRetriever
from rag_baseline import BaselineRAG, RAGRequest, RAGResponse, create_baseline_rag
from llm_client import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    app_name: str = "MAI Storage RAG API"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_url: str = "http://localhost:6333"
    
    class Config:
        env_file = "../../.env"
        extra = "ignore"  # Ignore extra fields in .env file


settings = Settings()

app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG API with hybrid retrieval, reranking, and MMR diversity",
    version="0.2.0",
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
retriever: Optional[HybridRetriever] = None
baseline_rag: Optional[BaselineRAG] = None
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global retriever, baseline_rag
    
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
            api_key=settings.openai_api_key
        )
        
        logger.info("Baseline RAG system initialized successfully")
        
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
            "endpoints": {
                "retrieve": "POST /retrieve - Hybrid document retrieval",
                "rag": "POST /rag - Baseline RAG generation with citations",
                "health": "GET /health - System health check",
                "stats": "GET /stats - Performance metrics"
            }
        },
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check(retriever_instance: HybridRetriever = Depends(get_retriever)):
    """Comprehensive health check."""
    uptime_seconds = time.time() - app_start_time
    
    # Check component health
    qdrant_healthy = True
    embeddings_healthy = True
    reranker_healthy = True
    
    try:
        # Test Qdrant connection
        retriever_instance.qdrant_client.get_collections()
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        qdrant_healthy = False
    
    try:
        # Test embeddings (quick test)
        await retriever_instance.embed_query("test")
    except Exception as e:
        logger.warning(f"Embeddings health check failed: {e}")
        embeddings_healthy = False
    
    try:
        # Test reranker model (lazy loading check)
        _ = retriever_instance.reranker_model
    except Exception as e:
        logger.warning(f"Reranker health check failed: {e}")
        reranker_healthy = False
    
    # Get stats
    stats = retriever_instance.get_stats()
    
    status = "healthy" if all([qdrant_healthy, embeddings_healthy, reranker_healthy]) else "unhealthy"
    
    return HealthCheck(
        status=status,
        timestamp=datetime.utcnow(),
        version="0.2.0",
        qdrant_healthy=qdrant_healthy,
        embeddings_healthy=embeddings_healthy,
        reranker_healthy=reranker_healthy,
        uptime_seconds=uptime_seconds,
        total_requests=stats.get("total_queries", 0)
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


@app.get("/stats", response_model=dict)
async def get_stats(retriever_instance: HybridRetriever = Depends(get_retriever)):
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
