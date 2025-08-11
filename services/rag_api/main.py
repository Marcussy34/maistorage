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
async def run_evaluation(
    mode: str = "traditional",  # "traditional", "agentic", or "both" 
    top_k: int = 5,
    save_results: bool = True
):
    """
    Run evaluation on Traditional or Agentic RAG using golden QA dataset.
    
    This endpoint implements Phase 8 evaluation harness with RAGAS metrics
    and retrieval-specific metrics (Recall@k, nDCG, MRR).
    
    Args:
        mode: Evaluation mode - "traditional", "agentic", or "both"
        top_k: Number of documents to retrieve for evaluation
        save_results: Whether to save results to disk
        
    Returns:
        Evaluation results with RAGAS metrics and performance data
    """
    try:
        from eval.run_ragas import RAGEvaluator
        import json
        from pathlib import Path
        
        logger.info(f"Starting evaluation - mode={mode}, top_k={top_k}")
        
        # Load golden QA dataset
        golden_qa_path = Path(__file__).parent / "golden_qa.json"
        with open(golden_qa_path, 'r') as f:
            golden_qa_data = json.load(f)
        
        questions = golden_qa_data["questions"]
        
        # Initialize RAG systems
        retriever_instance = get_retriever()
        baseline_rag_instance = get_baseline_rag()
        agentic_rag_instance = get_agentic_rag()
        
        evaluator = RAGEvaluator(
            baseline_rag=baseline_rag_instance,
            agentic_rag=agentic_rag_instance,
            retriever=retriever_instance
        )
        
        results = {}
        
        # Run Traditional RAG evaluation
        if mode in ["traditional", "both"]:
            logger.info("Running Traditional RAG evaluation...")
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
        
        # Run Agentic RAG evaluation  
        if mode in ["agentic", "both"]:
            logger.info("Running Agentic RAG evaluation...")
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
        
        # Add metadata
        results["metadata"] = {
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "top_k": top_k,
            "total_questions": len(questions),
            "golden_qa_version": golden_qa_data.get("dataset_info", {}).get("version", "unknown")
        }
        
        logger.info(f"Evaluation completed successfully - mode={mode}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="EVALUATION_FAILED", 
                    message=f"Evaluation failed: {str(e)}"
                )
            ).dict()
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
            return {"results": [], "message": "No evaluation results found"}
        
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
        
        return {
            "results": results,
            "total_files": len(results),
            "results_directory": str(results_dir)
        }
        
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
            return {
                "error": "Could not find both traditional and agentic evaluation results",
                "available_files": [f.name for f in results_dir.glob("*.json")]
            }
        
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
        
        return comparison
        
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
