"""
Edge case tests for MAI Storage RAG API.

Tests handling of ambiguous acronyms, typos, conflicting documents,
malformed inputs, and other challenging scenarios.
"""

import pytest
import asyncio
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

# Add parent directories to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import (
    RetrievalRequest, RetrievalResult,
    SentenceCitation, TextSpan
)
from retrieval import HybridRetriever
from rag_baseline import BaselineRAG
from graph import AgenticRAG
from tools import (
    tokenize_text, preprocess_text, validate_query,
    reciprocal_rank_fusion, calculate_bm25_scores
)


class TestAmbiguousAcronyms:
    """Test handling of ambiguous acronyms and abbreviations."""
    
    @pytest.fixture
    def mock_retriever_with_acronyms(self):
        """Mock retriever that returns conflicting acronym meanings."""
        retriever = Mock(spec=HybridRetriever)
        
        def mock_retrieve_with_acronyms(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            query = request.query if hasattr(request, 'query') else str(request)
            
            if "AI" in query or "ai" in query.lower():
                return [
                    RetrievalResult(
                        doc_id="ai_artificial_intelligence",
                        doc_name="ai_overview.md",
                        chunk_index=0,
                        text="AI stands for Artificial Intelligence, the simulation of human intelligence in machines.",
                        relevance_score=0.9,
                        rerank_score=0.87,
                        method_scores={"dense": 0.9, "bm25": 0.8}
                    ),
                    RetrievalResult(
                        doc_id="ai_adobe_illustrator",
                        doc_name="design_tools.md",
                        chunk_index=1,
                        text="AI is also the file extension for Adobe Illustrator, a vector graphics software.",
                        relevance_score=0.7,
                        rerank_score=0.65,
                        method_scores={"dense": 0.7, "bm25": 0.6}
                    ),
                    RetrievalResult(
                        doc_id="ai_action_item",
                        doc_name="project_management.md",
                        chunk_index=2,
                        text="In project management, AI can stand for Action Item, a task assigned to team members.",
                        relevance_score=0.6,
                        rerank_score=0.55,
                        method_scores={"dense": 0.6, "bm25": 0.5}
                    )
                ]
            return []
        
        retriever.retrieve = AsyncMock(side_effect=mock_retrieve_with_acronyms)
        return retriever
    
    @pytest.mark.asyncio
    async def test_ambiguous_acronym_context_resolution(self, mock_retriever_with_acronyms):
        """Test that context helps resolve ambiguous acronyms."""
        # Query with context that should disambiguate
        contextual_query = "What is AI in machine learning and deep learning applications?"
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="AI in machine learning refers to Artificial Intelligence, which involves creating systems that can learn and make decisions from data."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            from rag_baseline import RAGRequest
            baseline_rag = BaselineRAG(retriever=mock_retriever_with_acronyms)
            baseline_rag.llm = mock_llm
            
            request = RAGRequest(query=contextual_query, top_k=5)
            result = await baseline_rag.generate(request)
        
        # Should prioritize Artificial Intelligence meaning given context
        assert "artificial intelligence" in result["answer"].lower()
        assert len(result["sources"]) > 0
        
        # Should include the most relevant source
        top_source = result["sources"][0]
        assert "artificial intelligence" in top_source["text"].lower()
    
    @pytest.mark.asyncio
    async def test_acronym_without_context(self, mock_retriever_with_acronyms):
        """Test handling of ambiguous acronym without disambiguating context."""
        ambiguous_query = "What does AI mean?"
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="AI can have multiple meanings: Artificial Intelligence in technology, Adobe Illustrator in design, or Action Item in project management. The specific meaning depends on the context."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever_with_acronyms)
            baseline_rag.llm = mock_llm
            
            result = await baseline_rag.generate_response(
                query=ambiguous_query,
                top_k=5,
                include_sources=True
            )
        
        # Should acknowledge multiple meanings
        answer_lower = result["answer"].lower()
        assert "multiple" in answer_lower or "different" in answer_lower or "various" in answer_lower
        
        # Should include multiple sources representing different meanings
        assert len(result["sources"]) >= 2
    
    def test_acronym_tokenization(self):
        """Test that acronyms are properly tokenized."""
        text_with_acronyms = "The USA and UK have different AI policies. NASA's ML projects use GPU acceleration."
        
        tokens = tokenize_text(text_with_acronyms)
        
        # Acronyms should be preserved as tokens
        assert "usa" in tokens or "u.s.a" in tokens
        assert "uk" in tokens or "u.k" in tokens
        assert "ai" in tokens
        assert "nasa" in tokens or "nasa's" in tokens
        assert "ml" in tokens
        assert "gpu" in tokens
    
    def test_acronym_with_periods(self):
        """Test handling of acronyms with periods."""
        text = "The U.S.A. and U.K. collaborate on A.I. research through N.A.S.A."
        
        tokens = tokenize_text(text)
        processed = preprocess_text(text)
        
        # Should handle various acronym formats
        assert any("usa" in token.lower() or "u.s.a" in token.lower() for token in tokens)
        assert "u.s.a" in processed.lower() or "usa" in processed.lower()


class TestTyposAndMisspellings:
    """Test handling of typos and misspellings in queries."""
    
    @pytest.fixture
    def mock_retriever_fuzzy(self):
        """Mock retriever that can handle fuzzy matching."""
        retriever = Mock(spec=HybridRetriever)
        
        def mock_fuzzy_retrieve(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            query = request.query if hasattr(request, 'query') else str(request)
            
            # Simulate fuzzy matching for common misspellings
            if any(word in query.lower() for word in ["python", "pythong", "pyhton", "phyton"]):
                return [
                    RetrievalResult(
                        doc_id="python_doc",
                        doc_name="python_guide.md",
                        chunk_index=0,
                        text="Python is a high-level programming language known for its simplicity and readability.",
                        relevance_score=0.85,
                        rerank_score=0.8,
                        method_scores={"dense": 0.8, "bm25": 0.75}
                    )
                ]
            elif any(word in query.lower() for word in ["machine", "mashine", "machien"]):
                return [
                    RetrievalResult(
                        doc_id="ml_doc",
                        doc_name="machine_learning.md",
                        chunk_index=0,
                        text="Machine learning is a method of data analysis that automates analytical model building.",
                        relevance_score=0.82,
                        rerank_score=0.78,
                        method_scores={"dense": 0.8, "bm25": 0.7}
                    )
                ]
            return []
        
        retriever.retrieve = AsyncMock(side_effect=mock_fuzzy_retrieve)
        return retriever
    
    @pytest.mark.asyncio
    async def test_common_programming_typos(self, mock_retriever_fuzzy):
        """Test handling of common programming-related typos."""
        typo_queries = [
            "What is pythong programming?",  # pythong -> python
            "Explain pyhton syntax",         # pyhton -> python  
            "How does phyton work?",         # phyton -> python
        ]
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="Python is a programming language known for its readability and extensive libraries."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever_fuzzy)
            baseline_rag.llm = mock_llm
            
            for query in typo_queries:
                result = await baseline_rag.generate_response(
                    query=query,
                    top_k=5,
                    include_sources=True
                )
                
                # Should still find relevant information despite typos
                assert "python" in result["answer"].lower()
                assert len(result["sources"]) > 0
                assert "python" in result["sources"][0]["text"].lower()
    
    @pytest.mark.asyncio
    async def test_technical_term_misspellings(self, mock_retriever_fuzzy):
        """Test handling of misspelled technical terms."""
        misspelled_query = "What is mashine lerning and how does it work?"
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever_fuzzy)
            baseline_rag.llm = mock_llm
            
            result = await baseline_rag.generate_response(
                query=misspelled_query,
                top_k=5,
                include_sources=True
            )
        
        # Should correct to "machine learning"
        assert "machine learning" in result["answer"].lower()
        assert len(result["sources"]) > 0
    
    def test_typo_tokenization_robustness(self):
        """Test tokenization robustness with typos."""
        typo_text = "Pythong is grate for machien lerning and data analisys."
        
        tokens = tokenize_text(typo_text)
        
        # Should tokenize despite typos
        assert len(tokens) > 0
        assert "pythong" in tokens  # Preserve original for potential fuzzy matching
        assert "grate" in tokens
        assert "machien" in tokens
        assert "lerning" in tokens
        assert "analisys" in tokens


class TestConflictingDocuments:
    """Test handling of conflicting information across documents."""
    
    @pytest.fixture
    def mock_retriever_conflicting(self):
        """Mock retriever that returns conflicting information."""
        retriever = Mock(spec=HybridRetriever)
        
        def mock_conflicting_retrieve(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            query = request.query if hasattr(request, 'query') else str(request)
            
            if "python speed" in query.lower() or "python performance" in query.lower():
                return [
                    RetrievalResult(
                        doc_id="python_fast",
                        doc_name="python_advantages.md",
                        chunk_index=0,
                        text="Python is incredibly fast for development and has excellent performance with libraries like NumPy and Cython.",
                        relevance_score=0.9,
                        rerank_score=0.85,
                        method_scores={"dense": 0.9, "bm25": 0.8}
                    ),
                    RetrievalResult(
                        doc_id="python_slow", 
                        doc_name="python_limitations.md",
                        chunk_index=0,
                        text="Python is notoriously slow compared to compiled languages like C++ and Java due to its interpreted nature.",
                        relevance_score=0.88,
                        rerank_score=0.83,
                        method_scores={"dense": 0.85, "bm25": 0.82}
                    ),
                    RetrievalResult(
                        doc_id="python_moderate",
                        doc_name="python_balanced_view.md", 
                        chunk_index=0,
                        text="Python performance varies significantly depending on the use case and implementation, with trade-offs between development speed and execution speed.",
                        relevance_score=0.85,
                        rerank_score=0.8,
                        method_scores={"dense": 0.8, "bm25": 0.75}
                    )
                ]
            return []
        
        retriever.retrieve = AsyncMock(side_effect=mock_conflicting_retrieve)
        return retriever
    
    @pytest.mark.asyncio
    async def test_conflicting_information_synthesis(self, mock_retriever_conflicting):
        """Test synthesis of conflicting information."""
        conflicting_query = "Is Python fast or slow in terms of performance?"
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="Python's performance characteristics are nuanced. While it may be slower than compiled languages for computational tasks, it offers excellent development speed and can achieve good performance with optimized libraries. The answer depends on the specific use case and implementation."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever_conflicting)
            baseline_rag.llm = mock_llm
            
            result = await baseline_rag.generate_response(
                query=conflicting_query,
                top_k=5,
                include_sources=True
            )
        
        # Should acknowledge nuance and different perspectives
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in [
            "depends", "varies", "nuanced", "both", "however", "while"
        ])
        
        # Should include multiple conflicting sources
        assert len(result["sources"]) >= 2
        source_texts = [source["text"].lower() for source in result["sources"]]
        assert any("fast" in text for text in source_texts)
        assert any("slow" in text for text in source_texts)
    
    @pytest.mark.asyncio
    async def test_contradictory_facts(self, mock_retriever_conflicting):
        """Test handling of directly contradictory facts."""
        contradiction_query = "What is the definitive answer about Python performance?"
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="There isn't a single definitive answer about Python performance, as sources provide different perspectives. Some emphasize its speed advantages with certain libraries, while others highlight its limitations compared to compiled languages."
        )
        
        with patch('graph.LLMClient', return_value=mock_llm):
            agentic_rag = AgenticRAG(retriever=mock_retriever_conflicting)
            agentic_rag.llm = mock_llm
            
            result = await agentic_rag.run(
                query=contradiction_query,
                enable_verification=True
            )
        
        # Should acknowledge uncertainty and conflicting information
        answer_lower = result["answer"].lower()
        assert any(phrase in answer_lower for phrase in [
            "different perspectives", "conflicting", "varies", "not definitive"
        ])
    
    def test_conflicting_document_ranking(self):
        """Test that conflicting documents can be properly ranked."""
        # Simulate RRF with conflicting rankings
        rankings = [
            ["doc_fast", "doc_slow", "doc_moderate"],     # One ranking system
            ["doc_slow", "doc_moderate", "doc_fast"],     # Conflicting ranking
            ["doc_moderate", "doc_fast", "doc_slow"]      # Balanced ranking
        ]
        
        rrf_results = reciprocal_rank_fusion(rankings, k=60)
        
        # All documents should be included despite conflicts
        assert len(rrf_results) == 3
        
        # Results should be properly scored
        doc_scores = {doc_id: score for doc_id, score in rrf_results}
        assert all(score > 0 for score in doc_scores.values())


class TestMalformedInputs:
    """Test handling of malformed and invalid inputs."""
    
    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test handling of empty queries."""
        mock_retriever = Mock(spec=HybridRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[])
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="I need a specific question to provide a helpful answer."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm
            
            result = await baseline_rag.generate_response(
                query="",
                top_k=5,
                include_sources=True
            )
        
        # Should handle gracefully
        assert "answer" in result
        assert isinstance(result["answer"], str)
    
    @pytest.mark.asyncio
    async def test_extremely_long_query(self):
        """Test handling of extremely long queries."""
        # Create a very long query (>10k characters)
        long_query = "What is Python? " * 1000
        
        mock_retriever = Mock(spec=HybridRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[])
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="This query is very long. Python is a programming language."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm
            
            # Should handle without crashing
            result = await baseline_rag.generate_response(
                query=long_query,
                top_k=5,
                include_sources=True
            )
        
        assert "answer" in result
    
    @pytest.mark.asyncio
    async def test_special_character_queries(self):
        """Test queries with special characters and encoding."""
        special_queries = [
            "What is ¿Pythön? 🐍",  # Unicode and emoji
            "SELECT * FROM table; DROP TABLE users;",  # SQL injection attempt
            "../../etc/passwd",  # Path traversal attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "NULL\x00byte",  # Null byte
            "\\\\server\\share\\file",  # Windows path
        ]
        
        mock_retriever = Mock(spec=HybridRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[])
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="I can help you with programming questions."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm
            
            for query in special_queries:
                try:
                    result = await baseline_rag.generate_response(
                        query=query,
                        top_k=5,
                        include_sources=True
                    )
                    # Should handle all queries without crashing
                    assert "answer" in result
                except Exception as e:
                    # If any exception occurs, it should be a controlled one
                    assert "validation" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_invalid_query_validation(self):
        """Test query validation function."""
        # Test various invalid inputs
        invalid_queries = [
            None,
            123,
            [],
            {},
            "",
            "   ",  # Whitespace only
        ]
        
        for query in invalid_queries:
            try:
                is_valid = validate_query(query)
                # Should either return False or handle gracefully
                if isinstance(is_valid, bool):
                    assert is_valid is False
            except (TypeError, ValueError):
                # Acceptable to raise validation errors
                pass
    
    def test_malformed_retrieval_request(self):
        """Test handling of malformed retrieval requests."""
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            RetrievalRequest(
                query="test",
                top_k=-1,  # Invalid negative value
                collection_name=None
            )
        
        with pytest.raises((ValueError, TypeError)):
            RetrievalRequest(
                query="test",
                top_k="invalid",  # Wrong type
                collection_name="test"
            )


class TestUnicodeAndInternationalization:
    """Test handling of Unicode and international text."""
    
    def test_unicode_query_processing(self):
        """Test processing of Unicode queries."""
        unicode_queries = [
            "¿Qué es Python?",  # Spanish
            "Qu'est-ce que Python?",  # French
            "Was ist Python?",  # German
            "Pythonとは何ですか？",  # Japanese
            "Python是什么？",  # Chinese
            "Что такое Python?",  # Russian
            "파이썬이 무엇인가요?",  # Korean
        ]
        
        for query in unicode_queries:
            # Test tokenization
            tokens = tokenize_text(query)
            assert len(tokens) > 0
            
            # Test preprocessing
            processed = preprocess_text(query)
            assert isinstance(processed, str)
            assert len(processed) > 0
    
    def test_mixed_script_handling(self):
        """Test handling of mixed scripts and character sets."""
        mixed_text = "Python プログラミング 编程 프로그래밍 программирование"
        
        tokens = tokenize_text(mixed_text)
        processed = preprocess_text(mixed_text)
        
        # Should handle without crashing
        assert len(tokens) > 0
        assert len(processed) > 0
        assert "python" in tokens[0].lower()
    
    def test_emoji_and_symbols(self):
        """Test handling of emojis and special symbols."""
        emoji_text = "Python 🐍 is great! ✨ Machine learning 🤖 with AI 🧠"
        
        tokens = tokenize_text(emoji_text)
        processed = preprocess_text(emoji_text)
        
        # Should preserve meaningful text while handling emojis
        assert "python" in tokens
        assert "machine" in tokens
        assert "learning" in tokens
        
        # Emojis might be filtered or preserved depending on implementation
        assert len(processed) > 0


class TestEdgeCaseIntegration:
    """Integration tests combining multiple edge cases."""
    
    @pytest.mark.asyncio
    async def test_multiple_edge_cases_combined(self):
        """Test handling of queries with multiple edge case characteristics."""
        complex_query = "Can u explian pythong AI 🤖 vs ML mashines lerning  difference??? <script>"
        
        # This query has:
        # - Typos: "explian", "pythong", "mashines", "lerning"  
        # - Informal language: "u"
        # - Ambiguous acronyms: "AI", "ML"
        # - Emojis: 🤖
        # - Multiple punctuation: ???
        # - Potential XSS: <script>
        
        mock_retriever = Mock(spec=HybridRetriever)
        mock_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(
                doc_id="ai_ml_comparison",
                doc_name="ai_vs_ml.md",
                chunk_index=0,
                text="AI (Artificial Intelligence) is a broader concept that includes ML (Machine Learning) as a subset. Python is commonly used for both AI and ML development.",
                relevance_score=0.8,
                rerank_score=0.75,
                method_scores={"dense": 0.8, "bm25": 0.7}
            )
        ])
        
        mock_llm = Mock()
        mock_llm.generate_text = AsyncMock(
            return_value="AI (Artificial Intelligence) and ML (Machine Learning) are related but distinct concepts. AI is the broader field of creating intelligent machines, while ML is a specific approach within AI that focuses on algorithms that improve through experience. Python is a popular language for both AI and ML development."
        )
        
        with patch('rag_baseline.LLMClient', return_value=mock_llm):
            baseline_rag = BaselineRAG(retriever=mock_retriever)
            baseline_rag.llm = mock_llm
            
            result = await baseline_rag.generate_response(
                query=complex_query,
                top_k=5,
                include_sources=True
            )
        
        # Should handle all edge cases and provide coherent response
        assert "answer" in result
        answer_lower = result["answer"].lower()
        assert "ai" in answer_lower or "artificial intelligence" in answer_lower
        assert "ml" in answer_lower or "machine learning" in answer_lower
        assert "python" in answer_lower
        
        # Should not include malicious content
        assert "<script>" not in result["answer"]


if __name__ == "__main__":
    pytest.main([__file__])
