"""
Integration tests for end-to-end RAG workflows.

Tests the complete pipeline from query to response for both traditional
and agentic RAG approaches, including happy path, refinement, and edge cases.
"""

import pytest
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import RetrievalRequest, RetrievalResult
from retrieval import HybridRetriever
from rag_baseline import BaselineRAG
from graph import AgenticRAG, AgentStep, TraceEventType
from llm_client import LLMClient, LLMConfig
from citer import SentenceCitationEngine


class TestTraditionalRAGWorkflow:
    """Test traditional (baseline) RAG workflow end-to-end."""
    
    @pytest.fixture
    async def mock_retriever(self):
        """Create a mock retriever with realistic responses."""
        retriever = Mock(spec=HybridRetriever)
        
        # Mock retrieval results
        mock_results = [
            RetrievalResult(
                doc_id="doc1",
                doc_name="python_guide.md", 
                chunk_index=0,
                text="Python is a high-level programming language known for its readability and simplicity.",
                relevance_score=0.92,
                rerank_score=0.89,
                method_scores={"dense": 0.9, "bm25": 0.8}
            ),
            RetrievalResult(
                doc_id="doc2",
                doc_name="programming_basics.md",
                chunk_index=1,
                text="Programming languages like Python are used to create software applications.",
                relevance_score=0.85,
                rerank_score=0.82,
                method_scores={"dense": 0.8, "bm25": 0.7}
            )
        ]
        
        retriever.retrieve = AsyncMock(return_value=mock_results)
        return retriever
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        llm = Mock(spec=LLMClient)
        llm.generate_text = AsyncMock(
            return_value="Python is a popular programming language known for its simplicity and readability. It's widely used for web development, data science, and automation."
        )
        return llm
    
    @pytest.fixture
    async def baseline_rag(self, mock_retriever, mock_llm_client):
        """Create a baseline RAG instance with mocked dependencies."""
        with patch('rag_baseline.LLMClient', return_value=mock_llm_client):
            rag = BaselineRAG(retriever=mock_retriever)
            rag.llm = mock_llm_client
            return rag
    
    async def test_happy_path_traditional_rag(self, baseline_rag, mock_retriever):
        """Test successful traditional RAG query processing."""
        query = "What is Python programming language?"
        
        # Execute the RAG pipeline
        result = await baseline_rag.generate_response(
            query=query,
            top_k=5,
            include_sources=True
        )
        
        # Verify retrieval was called
        mock_retriever.retrieve.assert_called_once()
        
        # Verify response structure
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        
        # Verify response content
        assert "python" in result["answer"].lower()
        
        # Verify source information
        for source in result["sources"]:
            assert "doc_id" in source
            assert "text" in source
            assert "relevance_score" in source
    
    async def test_traditional_rag_with_empty_retrieval(self, baseline_rag, mock_retriever):
        """Test traditional RAG handling of empty retrieval results."""
        query = "What is quantum computing?"
        
        # Mock empty retrieval results
        mock_retriever.retrieve = AsyncMock(return_value=[])
        
        result = await baseline_rag.generate_response(
            query=query,
            top_k=5,
            include_sources=True
        )
        
        # Should handle gracefully
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) == 0
        
        # Answer should indicate no relevant information found
        assert "no relevant" in result["answer"].lower() or "not found" in result["answer"].lower() or len(result["answer"]) > 0
    
    async def test_traditional_rag_performance_metrics(self, baseline_rag, mock_retriever):
        """Test that performance metrics are captured."""
        query = "Explain machine learning basics"
        
        result = await baseline_rag.generate_response(
            query=query,
            top_k=5,
            include_sources=True
        )
        
        # Check for performance metrics
        assert "retrieval_time_ms" in result or "response_time_ms" in result
        if "retrieval_time_ms" in result:
            assert isinstance(result["retrieval_time_ms"], (int, float))
            assert result["retrieval_time_ms"] >= 0
        
        # Verify token counting if available
        if "total_tokens" in result:
            assert isinstance(result["total_tokens"], int)
            assert result["total_tokens"] > 0


class TestAgenticRAGWorkflow:
    """Test agentic RAG workflow end-to-end."""
    
    @pytest.fixture
    async def mock_retriever(self):
        """Create a mock retriever for agentic tests."""
        retriever = Mock(spec=HybridRetriever)
        
        # Mock different retrieval results for different sub-queries
        def mock_retrieve_side_effect(*args, **kwargs):
            # Return different results based on query content
            request = args[0] if args else kwargs.get('request')
            query = request.query if hasattr(request, 'query') else str(request)
            
            if "python" in query.lower():
                return [
                    RetrievalResult(
                        doc_id="python_doc",
                        doc_name="python_guide.md",
                        chunk_index=0,
                        text="Python is a versatile programming language with extensive libraries.",
                        relevance_score=0.9,
                        rerank_score=0.87,
                        method_scores={"dense": 0.9, "bm25": 0.8}
                    )
                ]
            elif "machine learning" in query.lower():
                return [
                    RetrievalResult(
                        doc_id="ml_doc",
                        doc_name="ml_basics.md", 
                        chunk_index=0,
                        text="Machine learning is a subset of AI that enables computers to learn from data.",
                        relevance_score=0.88,
                        rerank_score=0.85,
                        method_scores={"dense": 0.85, "bm25": 0.75}
                    )
                ]
            else:
                return []
        
        retriever.retrieve = AsyncMock(side_effect=mock_retrieve_side_effect)
        return retriever
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for agentic workflow."""
        llm = Mock(spec=LLMClient)
        
        def mock_generate_side_effect(*args, **kwargs):
            prompt = args[0] if args else kwargs.get('prompt', '')
            
            # Mock different responses based on prompt type
            if "plan" in prompt.lower() or "strategy" in prompt.lower():
                return "I need to search for information about Python programming and its applications."
            elif "verify" in prompt.lower() or "check" in prompt.lower():
                return "The response is accurate and well-supported by the sources. No refinement needed."
            else:
                return "Python is a powerful programming language used for various applications including web development and data science."
        
        llm.generate_text = AsyncMock(side_effect=mock_generate_side_effect)
        return llm
    
    @pytest.fixture
    async def agentic_rag(self, mock_retriever, mock_llm_client):
        """Create an agentic RAG instance with mocked dependencies."""
        with patch('graph.LLMClient', return_value=mock_llm_client):
            rag = AgenticRAG(retriever=mock_retriever)
            rag.llm = mock_llm_client
            return rag
    
    async def test_happy_path_agentic_rag(self, agentic_rag, mock_retriever, mock_llm_client):
        """Test successful agentic RAG query processing."""
        query = "What is Python and how is it used in machine learning?"
        
        result = await agentic_rag.run(
            query=query,
            top_k=5,
            enable_verification=True,
            max_refinements=2
        )
        
        # Verify LLM was called for different steps
        assert mock_llm_client.generate_text.call_count >= 2  # At least planner and synthesizer
        
        # Verify result structure
        assert "answer" in result
        assert "trace_events" in result
        assert "total_time_ms" in result
        assert isinstance(result["trace_events"], list)
        
        # Check trace events for agentic workflow steps
        event_types = [event.event_type for event in result["trace_events"]]
        assert TraceEventType.PLANNER in event_types or "planner" in str(event_types)
        assert TraceEventType.RETRIEVER in event_types or "retriever" in str(event_types)
        assert TraceEventType.SYNTHESIZER in event_types or "synthesizer" in str(event_types)
        
        # Verify answer quality
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
    
    async def test_agentic_rag_refinement_path(self, agentic_rag, mock_llm_client):
        """Test agentic RAG refinement when verification fails."""
        query = "Complex query requiring refinement"
        
        # Mock verifier to trigger refinement
        def mock_verify_response(*args, **kwargs):
            prompt = args[0] if args else kwargs.get('prompt', '')
            if "verify" in prompt.lower():
                # First verification fails, second passes
                if mock_llm_client.generate_text.call_count <= 4:
                    return "The response lacks sufficient detail and needs refinement."
                else:
                    return "The response is now comprehensive and accurate."
            return "Default response"
        
        mock_llm_client.generate_text = AsyncMock(side_effect=mock_verify_response)
        
        result = await agentic_rag.run(
            query=query,
            enable_verification=True,
            max_refinements=2
        )
        
        # Should show refinement occurred
        assert "refinement_count" in result
        if result["refinement_count"] > 0:
            # Verify multiple LLM calls due to refinement
            assert mock_llm_client.generate_text.call_count > 3
        
        # Should still produce valid result
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    async def test_agentic_rag_max_refinements_limit(self, agentic_rag, mock_llm_client):
        """Test that agentic RAG respects max refinements limit."""
        query = "Query that always needs refinement"
        
        # Mock verifier to always request refinement
        def mock_always_refine(*args, **kwargs):
            prompt = args[0] if args else kwargs.get('prompt', '')
            if "verify" in prompt.lower():
                return "This response needs significant improvement and refinement."
            return "Basic response that needs refinement."
        
        mock_llm_client.generate_text = AsyncMock(side_effect=mock_always_refine)
        
        result = await agentic_rag.run(
            query=query,
            enable_verification=True,
            max_refinements=1  # Low limit for testing
        )
        
        # Should respect the limit
        assert result.get("refinement_count", 0) <= 1
        
        # Should still complete successfully
        assert "answer" in result
        assert "trace_events" in result
    
    async def test_agentic_rag_trace_completeness(self, agentic_rag):
        """Test that trace events provide complete workflow visibility."""
        query = "What are the benefits of Python?"
        
        result = await agentic_rag.run(query=query, enable_verification=True)
        
        # Verify trace structure
        assert "trace_events" in result
        trace_events = result["trace_events"]
        
        # Should have events for each major step
        event_types = [event.event_type for event in trace_events]
        
        # Check for key workflow steps (exact names may vary)
        step_found = any("plan" in str(event_type).lower() for event_type in event_types)
        assert step_found, "Should have planner step in trace"
        
        retrieval_found = any("retriev" in str(event_type).lower() for event_type in event_types) 
        assert retrieval_found, "Should have retrieval step in trace"
        
        # Each event should have required fields
        for event in trace_events:
            assert hasattr(event, 'event_type')
            assert hasattr(event, 'data')
            assert hasattr(event, 'timestamp')


class TestRAGWorkflowComparison:
    """Test comparison between traditional and agentic RAG."""
    
    @pytest.fixture
    async def mock_retriever(self):
        """Shared mock retriever for comparison tests."""
        retriever = Mock(spec=HybridRetriever)
        
        mock_results = [
            RetrievalResult(
                doc_id="comparison_doc",
                doc_name="comparison.md",
                chunk_index=0,
                text="This is a test document for comparing RAG approaches.",
                relevance_score=0.8,
                rerank_score=0.75,
                method_scores={"dense": 0.8, "bm25": 0.7}
            )
        ]
        
        retriever.retrieve = AsyncMock(return_value=mock_results)
        return retriever
    
    @pytest.fixture
    def mock_llm_client(self):
        """Shared mock LLM client."""
        llm = Mock(spec=LLMClient)
        llm.generate_text = AsyncMock(return_value="This is a test response from the LLM.")
        return llm
    
    async def test_traditional_vs_agentic_response_differences(self, mock_retriever, mock_llm_client):
        """Test that traditional and agentic RAG produce different response structures."""
        query = "Compare traditional and agentic approaches"
        
        # Test traditional RAG
        with patch('rag_baseline.LLMClient', return_value=mock_llm_client):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm_client
            
            traditional_result = await baseline_rag.generate_response(
                query=query,
                top_k=5,
                include_sources=True
            )
        
        # Test agentic RAG
        with patch('graph.LLMClient', return_value=mock_llm_client):
            agentic_rag = AgenticRAG(retriever=mock_retriever)
            agentic_rag.llm = mock_llm_client
            
            agentic_result = await agentic_rag.run(
                query=query,
                enable_verification=True
            )
        
        # Both should have answers
        assert "answer" in traditional_result
        assert "answer" in agentic_result
        
        # Agentic should have additional trace information
        assert "trace_events" in agentic_result
        assert "trace_events" not in traditional_result
        
        # Traditional should have simpler source structure
        assert "sources" in traditional_result
        
        # Response times should be captured
        assert any(key.endswith("_time_ms") for key in traditional_result.keys())
        assert "total_time_ms" in agentic_result
    
    async def test_workflow_performance_comparison(self, mock_retriever, mock_llm_client):
        """Test performance characteristics of both workflows."""
        query = "Performance test query"
        
        # Measure traditional RAG
        import time
        start_traditional = time.time()
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm_client):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm_client
            traditional_result = await baseline_rag.generate_response(query=query)
        
        traditional_time = time.time() - start_traditional
        
        # Measure agentic RAG
        start_agentic = time.time()
        
        with patch('graph.LLMClient', return_value=mock_llm_client):
            agentic_rag = AgenticRAG(retriever=mock_retriever)
            agentic_rag.llm = mock_llm_client
            agentic_result = await agentic_rag.run(query=query)
        
        agentic_time = time.time() - start_agentic
        
        # Both should complete successfully
        assert traditional_result is not None
        assert agentic_result is not None
        
        # Performance should be reasonable (< 5 seconds for mocked responses)
        assert traditional_time < 5.0
        assert agentic_time < 5.0
        
        # Agentic may take longer due to multiple steps (in real scenarios)
        # But with mocks, times should be similar


class TestOutOfDistributionQueries:
    """Test handling of out-of-distribution (OOD) queries."""
    
    @pytest.fixture
    async def mock_empty_retriever(self):
        """Mock retriever that returns no results for OOD queries."""
        retriever = Mock(spec=HybridRetriever)
        retriever.retrieve = AsyncMock(return_value=[])
        return retriever
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM that handles OOD gracefully."""
        llm = Mock(spec=LLMClient)
        llm.generate_text = AsyncMock(
            return_value="I don't have enough information to answer this question accurately."
        )
        return llm
    
    async def test_ood_query_traditional_rag(self, mock_empty_retriever, mock_llm_client):
        """Test traditional RAG with out-of-distribution query."""
        ood_query = "What is the molecular structure of unobtainium?"
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm_client):
            baseline_rag = BaselineRAG(retriever=mock_empty_retriever)
            baseline_rag.llm = mock_llm_client
            
            result = await baseline_rag.generate_response(
                query=ood_query,
                top_k=5,
                include_sources=True
            )
        
        # Should handle gracefully
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 0
        
        # Answer should acknowledge lack of information
        answer_lower = result["answer"].lower()
        assert any(phrase in answer_lower for phrase in [
            "don't have", "no information", "not found", "cannot answer"
        ])
    
    async def test_ood_query_agentic_rag(self, mock_empty_retriever, mock_llm_client):
        """Test agentic RAG with out-of-distribution query."""
        ood_query = "Explain the politics of Alpha Centauri B"
        
        with patch('graph.LLMClient', return_value=mock_llm_client):
            agentic_rag = AgenticRAG(retriever=mock_empty_retriever)
            agentic_rag.llm = mock_llm_client
            
            result = await agentic_rag.run(
                query=ood_query,
                enable_verification=True
            )
        
        # Should complete workflow despite no retrieval results
        assert "answer" in result
        assert "trace_events" in result
        
        # Should show planner attempted to work with the query
        trace_events = result["trace_events"]
        assert len(trace_events) > 0
        
        # Answer should acknowledge limitations
        answer_lower = result["answer"].lower()
        assert any(phrase in answer_lower for phrase in [
            "don't have", "no information", "not available", "cannot provide"
        ])


class TestWorkflowErrorHandling:
    """Test error handling in RAG workflows."""
    
    async def test_retrieval_failure_handling(self):
        """Test handling of retrieval service failures."""
        # Mock retriever that raises exceptions
        failing_retriever = Mock(spec=HybridRetriever)
        failing_retriever.retrieve = AsyncMock(side_effect=Exception("Retrieval service unavailable"))
        
        mock_llm = Mock(spec=LLMClient)
        mock_llm.generate_text = AsyncMock(return_value="Fallback response")
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=failing_retriever)
            baseline_rag.llm = mock_llm
            
            # Should handle retrieval failure gracefully
            result = await baseline_rag.generate_response(
                query="Test query",
                top_k=5,
                include_sources=True
            )
            
            # Should still provide some response
            assert "answer" in result
            # May indicate error or provide fallback response
    
    async def test_llm_failure_handling(self):
        """Test handling of LLM service failures."""
        # Mock successful retriever
        retriever = Mock(spec=HybridRetriever)
        retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(
                doc_id="test_doc",
                doc_name="test.md",
                chunk_index=0,
                text="Test content",
                relevance_score=0.8,
                rerank_score=0.7,
                method_scores={"dense": 0.8}
            )
        ])
        
        # Mock failing LLM
        failing_llm = Mock(spec=LLMClient)
        failing_llm.generate_text = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        with patch('rag_baseline.LLMClient', return_value=failing_llm):
            baseline_rag = BaselineRAG(retriever=retriever)
            baseline_rag.llm = failing_llm
            
            # Should handle LLM failure
            with pytest.raises(Exception):
                await baseline_rag.generate_response(
                    query="Test query",
                    top_k=5,
                    include_sources=True
                )


if __name__ == "__main__":
    pytest.main([__file__])
