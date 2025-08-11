"""
Baseline RAG implementation for MAI Storage (Phase 3).

This module implements traditional single-pass RAG:
1. Query → Retrieve top-k chunks
2. Pack context → LLM generate answer
3. Return answer with chunk-level citations

Uses the hybrid retrieval system from Phase 2 and OpenAI chat completions.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

# Local imports
from models import (
    RetrievalRequest, RetrievalResponse, RetrievalResult, Document,
    RetrievalMethod, RerankMethod
)
from retrieval import HybridRetriever
from llm_client import LLMClient, LLMConfig, LLMResponse
from prompts.baseline import format_baseline_prompt, format_context_from_results

logger = logging.getLogger(__name__)


class RAGRequest(BaseModel):
    """Request for baseline RAG generation."""
    
    query: str = Field(..., description="User question", min_length=1)
    
    # Retrieval parameters  
    top_k: int = Field(default=10, description="Number of chunks to retrieve", ge=1, le=50)
    retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.HYBRID, description="Retrieval method")
    rerank_method: RerankMethod = Field(default=RerankMethod.BGE_RERANKER_V2, description="Reranking method")
    enable_mmr: bool = Field(default=True, description="Enable MMR diversity")
    
    # Generation parameters
    temperature: float = Field(default=0.7, description="LLM temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, description="Max tokens to generate")
    
    # Collection parameters
    collection_name: str = Field(default="maistorage_documents", description="Qdrant collection")
    
    # Output control
    include_context: bool = Field(default=True, description="Include retrieved context in response")
    include_citations: bool = Field(default=True, description="Include chunk-level citations")


class Citation(BaseModel):
    """Citation information for a source chunk."""
    
    document_id: str = Field(..., description="Document identifier")
    doc_name: Optional[str] = Field(None, description="Document name")
    chunk_index: Optional[int] = Field(None, description="Chunk index")
    score: float = Field(..., description="Relevance score")
    text_snippet: str = Field(..., description="Relevant text snippet")


class RAGResponse(BaseModel):
    """Response from baseline RAG generation."""
    
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer")
    
    # Citations and sources
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    context_used: Optional[str] = Field(None, description="Full context provided to LLM")
    
    # Performance metrics
    retrieval_time_ms: float = Field(..., description="Time spent on retrieval")
    generation_time_ms: float = Field(..., description="Time spent on LLM generation")
    total_time_ms: float = Field(..., description="Total processing time")
    
    # Model info
    model_used: str = Field(..., description="LLM model used for generation")
    tokens_used: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    
    # Retrieval info
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    retrieval_method: RetrievalMethod = Field(..., description="Retrieval method used")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class BaselineRAG:
    """
    Traditional single-pass RAG implementation.
    
    Combines hybrid retrieval from Phase 2 with LLM generation to provide
    coherent answers with proper citations.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize baseline RAG system.
        
        Args:
            retriever: Hybrid retriever instance from Phase 2
            llm_client: Optional pre-configured LLM client
            llm_config: LLM configuration if creating new client
        """
        self.retriever = retriever
        
        # Set up LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            config = llm_config or LLMConfig()
            self.llm_client = LLMClient(config)
        
        # Performance tracking
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
        
        logger.info("Baseline RAG system initialized")
    
    async def generate(self, request: RAGRequest) -> RAGResponse:
        """
        Generate answer using baseline RAG approach.
        
        Args:
            request: RAG request with query and parameters
            
        Returns:
            RAG response with answer and citations
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            retrieval_request = RetrievalRequest(
                query=request.query,
                method=request.retrieval_method,
                top_k=request.top_k,
                rerank_method=request.rerank_method,
                enable_mmr=request.enable_mmr,
                collection_name=request.collection_name,
                include_metadata=True,
                include_scores=True
            )
            
            retrieval_response = await self.retriever.retrieve(retrieval_request)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            
            # Step 2: Format context and create prompt
            context = format_context_from_results(retrieval_response.results)
            messages = format_baseline_prompt(request.query, context)
            
            # Step 3: Generate answer with LLM
            generation_start = time.time()
            llm_response = await self.llm_client.achat_completion(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Step 4: Create citations
            citations = self._create_citations(retrieval_response.results)
            
            # Step 5: Build response
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.total_queries += 1
            self.total_retrieval_time += retrieval_time_ms
            self.total_generation_time += generation_time_ms
            
            response = RAGResponse(
                query=request.query,
                answer=llm_response.content,
                citations=citations,
                context_used=context if request.include_context else None,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=total_time_ms,
                model_used=llm_response.model,
                tokens_used=llm_response.usage,
                chunks_retrieved=len(retrieval_response.results),
                retrieval_method=retrieval_response.method_used
            )
            
            logger.info(
                f"RAG completed: {len(citations)} sources, "
                f"{llm_response.usage.get('total_tokens', 0)} tokens, "
                f"{total_time_ms:.2f}ms total"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            raise
    
    def _create_citations(self, results: List[RetrievalResult]) -> List[Citation]:
        """
        Create citation objects from retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of citation objects
        """
        citations = []
        
        for result in results:
            doc = result.document
            
            # Get the best available score
            score = (
                result.final_score or 
                result.rerank_score or 
                result.hybrid_score or 
                result.dense_score or 
                result.bm25_score or 
                0.0
            )
            
            # Create text snippet (first 200 chars)
            snippet = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            
            citation = Citation(
                document_id=doc.id,
                doc_name=doc.doc_name,
                chunk_index=doc.chunk_index,
                score=score,
                text_snippet=snippet
            )
            
            citations.append(citation)
        
        return citations
    
    def test_generation(self, test_query: str = "What is MAI Storage?") -> RAGResponse:
        """
        Test the RAG system with a simple query.
        
        Args:
            test_query: Query to test with
            
        Returns:
            Test RAG response
        """
        request = RAGRequest(
            query=test_query,
            top_k=5,
            temperature=0.3,
            max_tokens=200
        )
        
        # Run async function in sync context
        return asyncio.run(self.generate(request))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_retrieval_time = (
            self.total_retrieval_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        avg_generation_time = (
            self.total_generation_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        stats = {
            "total_queries": self.total_queries,
            "avg_retrieval_time_ms": round(avg_retrieval_time, 2),
            "avg_generation_time_ms": round(avg_generation_time, 2),
            "avg_total_time_ms": round(avg_retrieval_time + avg_generation_time, 2)
        }
        
        # Add LLM stats
        llm_stats = self.llm_client.get_stats()
        stats.update({"llm_" + k: v for k, v in llm_stats.items()})
        
        return stats


# Factory function for easy setup
def create_baseline_rag(
    retriever: HybridRetriever,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs
) -> BaselineRAG:
    """
    Factory function to create baseline RAG with common configurations.
    
    Args:
        retriever: Hybrid retriever instance
        model: OpenAI model name
        temperature: Default temperature for generation
        **kwargs: Additional LLM configuration parameters
        
    Returns:
        Configured BaselineRAG instance
    """
    llm_config = LLMConfig(
        model=model,
        temperature=temperature,
        **kwargs
    )
    
    return BaselineRAG(retriever=retriever, llm_config=llm_config)
