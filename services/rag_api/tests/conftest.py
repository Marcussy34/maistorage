"""
Global test configuration and fixtures for MAI Storage RAG API tests.

This file provides shared fixtures, test data, and configuration for all test modules.
"""

import pytest
import asyncio
import json
import os
import tempfile
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    RetrievalResult, RetrievalRequest, CitationEngineConfig,
    SentenceCitation, TextSpan, SentenceAttributionResult
)


# Test data fixtures
@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "doc_id": "python_basics",
            "doc_name": "python_guide.md",
            "content": "Python is a high-level programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "metadata": {
                "author": "Test Author",
                "date": "2023-01-01",
                "category": "programming"
            }
        },
        {
            "doc_id": "machine_learning",
            "doc_name": "ml_basics.md", 
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
            "metadata": {
                "author": "ML Expert",
                "date": "2023-02-01",
                "category": "ai"
            }
        },
        {
            "doc_id": "data_science",
            "doc_name": "data_science_intro.md",
            "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. Python and R are popular languages for data science.",
            "metadata": {
                "author": "Data Scientist",
                "date": "2023-03-01", 
                "category": "data"
            }
        }
    ]


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            doc_id="python_basics",
            doc_name="python_guide.md",
            chunk_index=0,
            text="Python is a high-level programming language known for its readability and simplicity.",
            relevance_score=0.92,
            rerank_score=0.89,
            method_scores={"dense": 0.9, "bm25": 0.8, "rerank": 0.89},
            metadata={"category": "programming", "difficulty": "beginner"}
        ),
        RetrievalResult(
            doc_id="machine_learning", 
            doc_name="ml_basics.md",
            chunk_index=1,
            text="Machine learning enables computers to learn from experience without explicit programming.",
            relevance_score=0.88,
            rerank_score=0.85,
            method_scores={"dense": 0.85, "bm25": 0.75, "rerank": 0.85},
            metadata={"category": "ai", "difficulty": "intermediate"}
        ),
        RetrievalResult(
            doc_id="data_science",
            doc_name="data_science_intro.md", 
            chunk_index=0,
            text="Data science uses scientific methods to extract knowledge from data. Python is popular for data science.",
            relevance_score=0.82,
            rerank_score=0.78,
            method_scores={"dense": 0.8, "bm25": 0.7, "rerank": 0.78},
            metadata={"category": "data", "difficulty": "intermediate"}
        )
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing various scenarios."""
    return {
        "simple": "What is Python?",
        "complex": "How does machine learning work and what are its applications in data science?",
        "ambiguous": "What is AI?",
        "technical": "Explain the difference between supervised and unsupervised learning algorithms",
        "typo": "What is machien lerning?",
        "empty": "",
        "very_long": "What is Python programming? " * 100,
        "unicode": "¿Qué es Python? 🐍",
        "special_chars": "What is <script>alert('test')</script> Python?",
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    import numpy as np
    
    return {
        "query_embedding": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "doc_embeddings": np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Very similar to query
            [0.2, 0.3, 0.4, 0.5, 0.6],  # Somewhat similar
            [0.9, 0.8, 0.7, 0.6, 0.5],  # Different but valid
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Zero vector
        ]),
        "sentence_embeddings": np.array([
            [0.15, 0.25, 0.35, 0.45, 0.55],
            [0.12, 0.22, 0.32, 0.42, 0.52],
            [0.18, 0.28, 0.38, 0.48, 0.58],
        ])
    }


@pytest.fixture
def sample_citations():
    """Sample citations for testing."""
    return [
        SentenceCitation(
            sentence="Python is a programming language.",
            sentence_index=0,
            source_document_id="python_basics",
            source_doc_name="python_guide.md",
            source_chunk_index=0,
            supporting_span=TextSpan(
                start_char=0,
                end_char=42,
                text="Python is a high-level programming language"
            ),
            attribution_score=0.95,
            attribution_method="cosine_similarity",
            confidence_level="high"
        ),
        SentenceCitation(
            sentence="Machine learning uses algorithms to analyze data.",
            sentence_index=1,
            source_document_id="machine_learning",
            source_doc_name="ml_basics.md",
            source_chunk_index=1,
            supporting_span=TextSpan(
                start_char=50,
                end_char=100,
                text="uses algorithms to analyze data, identify patterns"
            ),
            attribution_score=0.87,
            attribution_method="cosine_similarity",
            confidence_level="high"
        )
    ]


@pytest.fixture
def sample_attribution_result(sample_citations):
    """Sample sentence attribution result."""
    return SentenceAttributionResult(
        response_text="Python is a programming language. Machine learning uses algorithms to analyze data.",
        sentences=["Python is a programming language.", "Machine learning uses algorithms to analyze data."],
        sentence_citations=sample_citations,
        overall_confidence=0.91,
        sentences_with_citations=2,
        sentences_with_warnings=0,
        attribution_coverage=1.0,
        attribution_time_ms=150.0,
        unique_sources=["python_basics", "machine_learning"],
        source_usage_counts={"python_basics": 1, "machine_learning": 1}
    )


# Mock fixtures
@pytest.fixture
def mock_retriever():
    """Mock HybridRetriever for testing."""
    from retrieval import HybridRetriever
    
    retriever = Mock(spec=HybridRetriever)
    retriever.retrieve = AsyncMock()
    retriever.embed_text = AsyncMock()
    return retriever


@pytest.fixture  
def mock_llm_client():
    """Mock LLMClient for testing."""
    from llm_client import LLMClient
    
    llm = Mock(spec=LLMClient)
    llm.generate_text = AsyncMock()
    return llm


@pytest.fixture
def mock_citation_engine():
    """Mock SentenceCitationEngine for testing."""
    from citer import SentenceCitationEngine
    
    engine = Mock(spec=SentenceCitationEngine)
    engine.generate_sentence_citations = AsyncMock()
    return engine


# Configuration fixtures
@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "retrieval": {
            "top_k": 10,
            "rerank_top_k": 20,
            "mmr_lambda": 0.5,
            "rrf_k": 60
        },
        "llm": {
            "model": "gpt-4o-mini",
            "max_tokens": 1000,
            "temperature": 0.1
        },
        "citation": {
            "confidence_threshold": 0.7,
            "attribution_method": "cosine_similarity",
            "enable_llm_rephrasing": False
        },
        "performance": {
            "max_latency_ms": 5000,
            "max_memory_mb": 500,
            "min_throughput_rps": 10
        }
    }


@pytest.fixture
def citation_engine_config():
    """Citation engine configuration for testing."""
    return CitationEngineConfig(
        primary_attribution_method="cosine_similarity",
        confidence_threshold=0.7,
        enable_llm_rephrasing=False,
        max_sentence_length=200,
        min_attribution_score=0.5
    )


# Temporary file fixtures
@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_json_file(temp_directory):
    """Create a temporary JSON file for testing."""
    test_data = {
        "test_key": "test_value",
        "nested": {"key": "value"}
    }
    
    file_path = temp_directory / "test_data.json"
    with open(file_path, 'w') as f:
        json.dump(test_data, f)
    
    return file_path


# Async test utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data generators
@pytest.fixture
def large_text_generator():
    """Generator for large text content."""
    def generate(num_sentences: int = 1000) -> str:
        sentences = [
            f"This is sentence number {i} in a large text document."
            for i in range(num_sentences)
        ]
        return " ".join(sentences)
    
    return generate


@pytest.fixture
def retrieval_result_generator():
    """Generator for retrieval results."""
    def generate(count: int = 10, base_score: float = 0.8) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                doc_id=f"generated_doc_{i}",
                doc_name=f"generated_{i}.md",
                chunk_index=i % 5,
                text=f"This is generated content for document {i} with relevant information.",
                relevance_score=base_score - (i * 0.01),
                rerank_score=base_score - (i * 0.01) - 0.05,
                method_scores={
                    "dense": base_score - (i * 0.01),
                    "bm25": base_score - (i * 0.02),
                    "rerank": base_score - (i * 0.01) - 0.05
                },
                metadata={"generated": True, "index": i}
            )
            for i in range(count)
        ]
    
    return generate


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    test_env_vars = {
        "OPENAI_API_KEY": "test_key_12345",
        "OPENAI_MODEL": "gpt-4o-mini",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "QDRANT_URL": "http://localhost:6333",
        "TESTING": "true"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


# Test markers and configurations
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case test"
    )


# Skip conditions
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration."""
    if config.getoption("--run-slow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true", 
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run performance tests"
    )
