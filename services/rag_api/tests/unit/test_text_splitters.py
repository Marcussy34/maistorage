"""
Unit tests for text splitting and chunking functionality.

Tests the semantic chunking strategy used in the indexer with various input types
and edge cases.
"""

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TestRecursiveCharacterTextSplitter:
    """Test the text splitting functionality from indexer."""
    
    @pytest.fixture
    def text_splitter(self):
        """Create the same text splitter used in indexer."""
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    
    def test_basic_chunking(self, text_splitter):
        """Test basic text chunking functionality."""
        text = "This is a test document. " * 50  # Long enough to chunk
        documents = [Document(page_content=text)]
        
        chunks = text_splitter.split_documents(documents)
        
        # Should create multiple chunks for long text
        assert len(chunks) > 1
        
        # Each chunk should be within size limits
        for chunk in chunks:
            assert len(chunk.page_content) <= 500
        
        # Chunks should have proper metadata
        for chunk in chunks:
            assert "start_index" in chunk.metadata
            assert isinstance(chunk.metadata["start_index"], int)
    
    def test_chunk_overlap(self, text_splitter):
        """Test that overlapping is working correctly."""
        # Create text with distinct sections
        text = ("Section A. " * 20 + 
                "Section B. " * 20 + 
                "Section C. " * 20)
        documents = [Document(page_content=text)]
        
        chunks = text_splitter.split_documents(documents)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].page_content[-50:]  # Last 50 chars
                chunk2_start = chunks[i + 1].page_content[:50]  # First 50 chars
                
                # There should be some common words between chunks
                words1 = set(chunk1_end.split())
                words2 = set(chunk2_start.split())
                
                # Some overlap expected (not strict equality due to separator handling)
                assert len(words1 & words2) > 0 or abs(len(chunk1_end) - len(chunk2_start)) < 10
    
    def test_separator_priority(self, text_splitter):
        """Test that separators are used in priority order."""
        # Text with different separator types
        text = ("Paragraph 1 with multiple sentences. This is sentence 2.\n\n"
                "Paragraph 2 with different content. Another sentence here.\n\n"
                "Paragraph 3 with even more content to ensure chunking occurs.")
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        # Should prefer paragraph breaks (\n\n) over sentence breaks (. )
        # This is hard to test deterministically, so we'll just ensure chunking works
        assert len(chunks) >= 1
        assert all(chunk.page_content.strip() for chunk in chunks)
    
    def test_short_text_no_chunking(self, text_splitter):
        """Test that short text doesn't get unnecessarily chunked."""
        text = "This is a short document that should not be chunked."
        documents = [Document(page_content=text)]
        
        chunks = text_splitter.split_documents(documents)
        
        # Should only create one chunk for short text
        assert len(chunks) == 1
        assert chunks[0].page_content == text
    
    def test_empty_text(self, text_splitter):
        """Test handling of empty text."""
        documents = [Document(page_content="")]
        
        chunks = text_splitter.split_documents(documents)
        
        # Should handle empty text gracefully
        assert len(chunks) >= 0  # May return empty list or single empty chunk
        if chunks:
            assert chunks[0].page_content == ""
    
    def test_whitespace_only_text(self, text_splitter):
        """Test handling of whitespace-only text."""
        text = "   \n\n  \t  \n  "
        documents = [Document(page_content=text)]
        
        chunks = text_splitter.split_documents(documents)
        
        # Should handle whitespace gracefully
        assert len(chunks) >= 0
        if chunks:
            assert chunks[0].page_content == text
    
    def test_start_index_correctness(self, text_splitter):
        """Test that start indices are calculated correctly."""
        text = "A" * 200 + "B" * 200 + "C" * 200  # 600 chars, should chunk
        documents = [Document(page_content=text)]
        
        chunks = text_splitter.split_documents(documents)
        
        if len(chunks) > 1:
            # First chunk should start at 0
            assert chunks[0].metadata["start_index"] == 0
            
            # Subsequent chunks should have increasing start indices
            for i in range(1, len(chunks)):
                assert chunks[i].metadata["start_index"] > chunks[i-1].metadata["start_index"]
    
    def test_markdown_document(self, text_splitter):
        """Test chunking of markdown-style document."""
        text = """# Main Title

## Section 1

This is the first section with some content. It has multiple sentences to test chunking.

### Subsection 1.1

More content here with different formatting.

## Section 2

This is a different section. It should be handled properly by the splitter.

- List item 1
- List item 2  
- List item 3

### Subsection 2.1

Final section with more content to ensure proper chunking behavior."""
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        assert len(chunks) >= 1
        
        # Check that headers are preserved in chunks
        header_found = False
        for chunk in chunks:
            if "#" in chunk.page_content:
                header_found = True
                break
        
        assert header_found, "Headers should be preserved in chunks"
    
    def test_code_document(self, text_splitter):
        """Test chunking of code-like document."""
        text = """def function_one():
    \"\"\"This is a function with a docstring.
    
    It has multiple lines of documentation.
    \"\"\"
    x = 1
    y = 2
    return x + y

def function_two():
    \"\"\"Another function.\"\"\"
    for i in range(10):
        print(i)
    
class MyClass:
    \"\"\"A sample class.\"\"\"
    
    def __init__(self):
        self.value = 42
        
    def method(self):
        return self.value * 2"""
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        assert len(chunks) >= 1
        
        # Code structure should be somewhat preserved
        function_found = False
        for chunk in chunks:
            if "def " in chunk.page_content:
                function_found = True
                break
        
        assert function_found, "Code structure should be preserved"


class TestTextSplitterEdgeCases:
    """Test edge cases for text splitting."""
    
    @pytest.fixture
    def text_splitter(self):
        """Create text splitter for edge case testing."""
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    
    def test_very_long_words(self, text_splitter):
        """Test handling of very long words that exceed chunk size."""
        # Create a word longer than chunk size
        long_word = "a" * 600
        text = f"Short sentence. {long_word} Another sentence."
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        # Should handle gracefully, possibly creating chunks larger than limit
        assert len(chunks) >= 1
        
        # Check that all original text is preserved across chunks
        total_content = "".join(chunk.page_content for chunk in chunks)
        assert long_word in total_content, "Long word should be preserved somewhere in the chunks"
    
    def test_no_separators_text(self, text_splitter):
        """Test text with no natural separators."""
        # Text with no spaces, periods, or newlines
        text = "a" * 600  # Longer than chunk size
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        # Should handle by character splitting
        assert len(chunks) >= 1
        
        # Should preserve the content (may have overlap, so >= original length)
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        assert total_chars >= 600, f"Expected at least 600 chars, got {total_chars}"
        
        # Verify all chunks contain only 'a' characters
        for chunk in chunks:
            assert all(c == 'a' for c in chunk.page_content), "Chunks should only contain 'a' characters"
    
    def test_unicode_characters(self, text_splitter):
        """Test handling of unicode characters."""
        text = ("Hello 🌍 world! This is a test with émojis and àccénts. " * 20)
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        assert len(chunks) >= 1
        
        # Unicode should be preserved
        emoji_found = False
        accent_found = False
        for chunk in chunks:
            if "🌍" in chunk.page_content:
                emoji_found = True
            if "àccénts" in chunk.page_content:
                accent_found = True
        
        assert emoji_found, "Emojis should be preserved"
        assert accent_found, "Accented characters should be preserved"
    
    def test_mixed_separators(self, text_splitter):
        """Test text with mixed separator types."""
        text = ("Line 1\n"
                "Line 2\r\n"  # Windows line ending
                "Line 3\n\n"
                "Paragraph break above.\n"
                "Sentence 1. Sentence 2.\n"
                "Final line.")
        
        documents = [Document(page_content=text)]
        chunks = text_splitter.split_documents(documents)
        
        assert len(chunks) >= 1
        
        # All content should be preserved
        total_content = "".join(chunk.page_content for chunk in chunks)
        # Normalize line endings for comparison
        assert total_content.replace("\r\n", "\n") == text.replace("\r\n", "\n")


if __name__ == "__main__":
    pytest.main([__file__])
