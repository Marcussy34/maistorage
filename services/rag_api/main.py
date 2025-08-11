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

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from graph import AgenticRAG, create_agentic_rag, TraceEvent, TraceEventType

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
                            "verification_passed": not final_state.get("needs_refinement", False)
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
