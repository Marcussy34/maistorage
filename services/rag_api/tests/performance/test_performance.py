"""
Performance and load tests for MAI Storage RAG API.

Tests p50/p95 latency, memory usage, concurrent request handling,
and other performance characteristics.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directories to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import RetrievalRequest, RetrievalResult
from retrieval import HybridRetriever
from rag_baseline import BaselineRAG
from graph import AgenticRAG
from tools import (
    reciprocal_rank_fusion, maximal_marginal_relevance,
    calculate_bm25_scores, tokenize_text
)


class PerformanceMonitor:
    """Helper class to monitor performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
        self.measurements = []
    
    def measure(self, label: str = "measurement"):
        """Take a performance measurement."""
        current_time = time.time()
        current_memory = self.process.memory_info().rss
        
        measurement = {
            'label': label,
            'timestamp': current_time,
            'elapsed_time': current_time - self.start_time,
            'memory_rss': current_memory,
            'memory_delta': current_memory - self.start_memory,
            'cpu_percent': self.process.cpu_percent()
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.measurements:
            return {}
        
        memory_deltas = [m['memory_delta'] for m in self.measurements]
        elapsed_times = [m['elapsed_time'] for m in self.measurements]
        
        return {
            'total_time': max(elapsed_times),
            'peak_memory_delta': max(memory_deltas),
            'avg_memory_delta': statistics.mean(memory_deltas),
            'measurement_count': len(self.measurements),
            'memory_mb_delta': max(memory_deltas) / (1024 * 1024)
        }


class TestRetrievalPerformance:
    """Test performance of core retrieval algorithms."""
    
    def test_rrf_performance_large_rankings(self):
        """Test RRF performance with large ranking lists."""
        # Generate large rankings
        num_docs = 10000
        docs = [f'doc_{i}' for i in range(num_docs)]
        
        # Create multiple rankings with overlaps
        rankings = [
            docs[:5000],  # First 5000
            docs[2500:7500],  # Middle 5000
            docs[5000:],  # Last 5000
        ]
        
        monitor = PerformanceMonitor()
        start_time = time.time()
        
        # Test RRF performance
        results = reciprocal_rank_fusion(rankings, k=60)
        
        end_time = time.time()
        monitor.measure("rrf_large_rankings")
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"RRF took too long: {execution_time:.2f}s"
        
        # Verify correctness
        assert len(results) <= num_docs
        assert all(isinstance(score, float) for _, score in results)
        
        # Memory usage should be reasonable
        summary = monitor.get_summary()
        assert summary['memory_mb_delta'] < 100, f"Memory usage too high: {summary['memory_mb_delta']:.2f}MB"
    
    def test_mmr_performance_high_dimensions(self):
        """Test MMR performance with high-dimensional embeddings."""
        import numpy as np
        
        # High-dimensional embeddings (like OpenAI's 1536-dim)
        embedding_dim = 1536
        num_docs = 1000
        
        query_embedding = np.random.rand(embedding_dim)
        doc_embeddings = np.random.rand(num_docs, embedding_dim)
        doc_ids = [f'doc_{i}' for i in range(num_docs)]
        relevance_scores = np.random.rand(num_docs).tolist()
        
        monitor = PerformanceMonitor()
        start_time = time.time()
        
        # Test MMR performance
        results = maximal_marginal_relevance(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            doc_ids=doc_ids,
            relevance_scores=relevance_scores,
            lambda_param=0.5,
            top_k=50
        )
        
        end_time = time.time()
        monitor.measure("mmr_high_dim")
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"MMR took too long: {execution_time:.2f}s"
        
        # Verify correctness
        assert len(results) == 50
        assert all(isinstance(score, float) for _, score in results)
        
        # Memory usage check
        summary = monitor.get_summary()
        assert summary['memory_mb_delta'] < 200, f"Memory usage too high: {summary['memory_mb_delta']:.2f}MB"
    
    def test_bm25_performance_large_corpus(self):
        """Test BM25 performance with large document corpus."""
        # Generate large corpus
        num_docs = 5000
        avg_doc_length = 100
        
        query_tokens = ['machine', 'learning', 'python', 'data']
        doc_tokens_list = [
            [f'word_{i}_{j}' for j in range(avg_doc_length)] + ['machine', 'learning'] 
            for i in range(num_docs)
        ]
        
        monitor = PerformanceMonitor()
        start_time = time.time()
        
        # Test BM25 performance
        scores = calculate_bm25_scores(query_tokens, doc_tokens_list)
        
        end_time = time.time()
        monitor.measure("bm25_large_corpus")
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 3.0, f"BM25 took too long: {execution_time:.2f}s"
        
        # Verify correctness
        assert len(scores) == num_docs
        assert all(isinstance(score, float) for score in scores)
        
        # Memory usage check
        summary = monitor.get_summary()
        assert summary['memory_mb_delta'] < 150, f"Memory usage too high: {summary['memory_mb_delta']:.2f}MB"
    
    def test_tokenization_performance(self):
        """Test tokenization performance with large text."""
        # Generate large text document
        large_text = "This is a test sentence with multiple words. " * 10000
        
        monitor = PerformanceMonitor()
        start_time = time.time()
        
        # Test tokenization performance
        tokens = tokenize_text(large_text)
        
        end_time = time.time()
        monitor.measure("tokenization_large_text")
        
        # Performance assertions
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Tokenization took too long: {execution_time:.2f}s"
        
        # Verify correctness
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        
        # Memory usage should be reasonable
        summary = monitor.get_summary()
        assert summary['memory_mb_delta'] < 50, f"Memory usage too high: {summary['memory_mb_delta']:.2f}MB"


class TestRAGWorkflowPerformance:
    """Test performance of complete RAG workflows."""
    
    @pytest.fixture
    def mock_fast_retriever(self):
        """Mock retriever optimized for performance testing."""
        retriever = Mock(spec=HybridRetriever)
        
        # Fast mock responses
        mock_results = [
            RetrievalResult(
                doc_id=f"perf_doc_{i}",
                doc_name=f"perf_{i}.md",
                chunk_index=0,
                text=f"This is performance test document {i} with relevant content.",
                relevance_score=0.9 - (i * 0.1),
                rerank_score=0.85 - (i * 0.1),
                method_scores={"dense": 0.8, "bm25": 0.7}
            )
            for i in range(5)
        ]
        
        async def fast_retrieve(*args, **kwargs):
            # Simulate minimal processing time
            await asyncio.sleep(0.01)  # 10ms simulation
            return mock_results
        
        retriever.retrieve = fast_retrieve
        return retriever
    
    @pytest.fixture
    def mock_fast_llm(self):
        """Mock LLM optimized for performance testing."""
        llm = Mock()
        
        async def fast_generate(*args, **kwargs):
            # Simulate minimal generation time
            await asyncio.sleep(0.05)  # 50ms simulation
            return "This is a fast test response from the mock LLM for performance testing."
        
        llm.generate_text = fast_generate
        return llm
    
    async def test_traditional_rag_latency(self, mock_fast_retriever, mock_fast_llm):
        """Test traditional RAG latency characteristics."""
        with patch('rag_baseline.LLMClient', return_value=mock_fast_llm):
            baseline_rag = BaselineRAG(retriever=mock_fast_retriever)
            baseline_rag.llm = mock_fast_llm
            
            # Measure latency for multiple requests
            latencies = []
            num_requests = 20
            
            for i in range(num_requests):
                start_time = time.time()
                
                result = await baseline_rag.generate_response(
                    query=f"Test query {i}",
                    top_k=5,
                    include_sources=True
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Verify response
                assert "answer" in result
                assert "sources" in result
            
            # Calculate percentiles
            latencies.sort()
            p50 = statistics.median(latencies)
            p95 = latencies[int(0.95 * len(latencies))]
            p99 = latencies[int(0.99 * len(latencies))]
            
            # Performance assertions (adjust based on requirements)
            assert p50 < 200, f"P50 latency too high: {p50:.2f}ms"
            assert p95 < 500, f"P95 latency too high: {p95:.2f}ms"
            assert p99 < 1000, f"P99 latency too high: {p99:.2f}ms"
            
            print(f"Traditional RAG Latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
    
    async def test_agentic_rag_latency(self, mock_fast_retriever, mock_fast_llm):
        """Test agentic RAG latency characteristics."""
        with patch('graph.LLMClient', return_value=mock_fast_llm):
            agentic_rag = AgenticRAG(retriever=mock_fast_retriever)
            agentic_rag.llm = mock_fast_llm
            
            # Measure latency for multiple requests
            latencies = []
            num_requests = 10  # Fewer requests due to higher complexity
            
            for i in range(num_requests):
                start_time = time.time()
                
                result = await agentic_rag.run(
                    query=f"Complex test query {i}",
                    enable_verification=True,
                    max_refinements=1
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Verify response
                assert "answer" in result
                assert "trace_events" in result
            
            # Calculate percentiles
            latencies.sort()
            p50 = statistics.median(latencies)
            p95 = latencies[int(0.95 * len(latencies))]
            
            # Agentic RAG expected to be slower due to multiple steps
            assert p50 < 800, f"P50 latency too high: {p50:.2f}ms"
            assert p95 < 1500, f"P95 latency too high: {p95:.2f}ms"
            
            print(f"Agentic RAG Latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms")
    
    async def test_memory_usage_under_load(self, mock_fast_retriever, mock_fast_llm):
        """Test memory usage characteristics under load."""
        with patch('rag_baseline.LLMClient', return_value=mock_fast_llm):
            baseline_rag = BaselineRAG(retriever=mock_fast_retriever)
            baseline_rag.llm = mock_fast_llm
            
            monitor = PerformanceMonitor()
            initial_memory = monitor.process.memory_info().rss
            
            # Run multiple queries to test memory usage
            num_queries = 50
            
            for i in range(num_queries):
                await baseline_rag.generate_response(
                    query=f"Memory test query {i} with various content to test allocation",
                    top_k=10,
                    include_sources=True
                )
                
                if i % 10 == 0:  # Measure every 10 queries
                    monitor.measure(f"query_batch_{i}")
            
            final_memory = monitor.process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory usage should be reasonable
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f}MB"
            
            print(f"Memory increase after {num_queries} queries: {memory_increase:.2f}MB")


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    @pytest.fixture
    def mock_concurrent_retriever(self):
        """Mock retriever for concurrency testing."""
        retriever = Mock(spec=HybridRetriever)
        
        async def concurrent_retrieve(*args, **kwargs):
            # Simulate variable processing time
            await asyncio.sleep(0.02 + (hash(str(args)) % 10) * 0.001)
            return [
                RetrievalResult(
                    doc_id="concurrent_doc",
                    doc_name="concurrent.md",
                    chunk_index=0,
                    text="Concurrent retrieval test content.",
                    relevance_score=0.8,
                    rerank_score=0.75,
                    method_scores={"dense": 0.8, "bm25": 0.7}
                )
            ]
        
        retriever.retrieve = concurrent_retrieve
        return retriever
    
    @pytest.fixture
    def mock_concurrent_llm(self):
        """Mock LLM for concurrency testing."""
        llm = Mock()
        
        async def concurrent_generate(*args, **kwargs):
            # Simulate variable generation time
            await asyncio.sleep(0.03 + (hash(str(args)) % 10) * 0.002)
            return "Concurrent test response from mock LLM."
        
        llm.generate_text = concurrent_generate
        return llm
    
    async def test_concurrent_request_handling(self, mock_concurrent_retriever, mock_concurrent_llm):
        """Test handling of concurrent requests."""
        with patch('rag_baseline.LLMClient', return_value=mock_concurrent_llm):
            baseline_rag = BaselineRAG(retriever=mock_concurrent_retriever)
            baseline_rag.llm = mock_concurrent_llm
            
            # Create concurrent requests
            num_concurrent = 20
            
            async def single_request(request_id: int):
                start_time = time.time()
                result = await baseline_rag.generate_response(
                    query=f"Concurrent query {request_id}",
                    top_k=5,
                    include_sources=True
                )
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'latency_ms': (end_time - start_time) * 1000,
                    'success': "answer" in result
                }
            
            # Execute concurrent requests
            start_time = time.time()
            tasks = [single_request(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Verify all requests succeeded
            assert all(result['success'] for result in results)
            
            # Calculate throughput
            throughput = num_concurrent / total_time
            
            # Performance assertions
            latencies = [result['latency_ms'] for result in results]
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            assert throughput > 10, f"Throughput too low: {throughput:.2f} req/s"
            assert avg_latency < 200, f"Average latency too high: {avg_latency:.2f}ms"
            assert max_latency < 500, f"Max latency too high: {max_latency:.2f}ms"
            
            print(f"Concurrent Performance - Throughput: {throughput:.2f} req/s, Avg Latency: {avg_latency:.2f}ms")
    
    async def test_thread_safety(self, mock_concurrent_retriever, mock_concurrent_llm):
        """Test thread safety of RAG components."""
        with patch('rag_baseline.LLMClient', return_value=mock_concurrent_llm):
            baseline_rag = BaselineRAG(retriever=mock_concurrent_retriever)
            baseline_rag.llm = mock_concurrent_llm
            
            # Shared counter to test thread safety
            counter = {'value': 0}
            lock = threading.Lock()
            
            async def thread_safe_request(request_id: int):
                result = await baseline_rag.generate_response(
                    query=f"Thread safety test {request_id}",
                    top_k=3,
                    include_sources=True
                )
                
                # Update counter in thread-safe manner
                with lock:
                    counter['value'] += 1
                
                return result is not None
            
            # Run multiple concurrent requests
            num_threads = 15
            tasks = [thread_safe_request(i) for i in range(num_threads)]
            results = await asyncio.gather(*tasks)
            
            # Verify thread safety
            assert all(results), "Some requests failed"
            assert counter['value'] == num_threads, f"Counter mismatch: {counter['value']} != {num_threads}"


class TestStressAndLoadLimits:
    """Test system behavior under stress and at limits."""
    
    async def test_large_document_processing(self):
        """Test processing of very large documents."""
        # Create a very large text document
        large_content = "This is a test sentence. " * 50000  # ~1.25M characters
        
        monitor = PerformanceMonitor()
        start_time = time.time()
        
        # Test tokenization of large content
        tokens = tokenize_text(large_content)
        
        end_time = time.time()
        monitor.measure("large_document_tokenization")
        
        # Performance assertions
        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Large document processing too slow: {processing_time:.2f}s"
        
        # Verify correctness
        assert len(tokens) > 0
        
        # Memory usage check
        summary = monitor.get_summary()
        assert summary['memory_mb_delta'] < 200, f"Memory usage too high: {summary['memory_mb_delta']:.2f}MB"
    
    async def test_high_volume_queries(self):
        """Test system behavior with high volume of queries."""
        # Mock retriever with minimal overhead
        fast_retriever = Mock(spec=HybridRetriever)
        fast_retriever.retrieve = AsyncMock(return_value=[])
        
        fast_llm = Mock()
        fast_llm.generate_text = AsyncMock(return_value="Fast response")
        
        with patch('rag_baseline.LLMClient', return_value=fast_llm):
            baseline_rag = BaselineRAG(retriever=fast_retriever)
            baseline_rag.llm = fast_llm
            
            # High volume test
            num_queries = 100
            batch_size = 10
            
            total_start_time = time.time()
            
            for batch in range(0, num_queries, batch_size):
                batch_queries = [
                    baseline_rag.generate_response(
                        query=f"High volume query {i}",
                        top_k=5,
                        include_sources=True
                    )
                    for i in range(batch, min(batch + batch_size, num_queries))
                ]
                
                batch_results = await asyncio.gather(*batch_queries)
                
                # Verify batch results
                assert all("answer" in result for result in batch_results)
            
            total_time = time.time() - total_start_time
            throughput = num_queries / total_time
            
            # Performance assertion
            assert throughput > 50, f"High volume throughput too low: {throughput:.2f} queries/s"
            
            print(f"High Volume Performance - {throughput:.2f} queries/s for {num_queries} queries")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
