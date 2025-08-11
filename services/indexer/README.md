# MAI Storage Document Indexer

**Phase 1 Complete** ✅

This directory contains the document ingestion system for MAI Storage, implementing Phase 1 of the development plan.

## Overview

The document indexer processes raw documents (PDF, Markdown, HTML, TXT) and converts them into searchable vector embeddings stored in Qdrant. It implements semantic chunking with configurable overlap and comprehensive metadata tracking.

## Features

- **Multi-format Support**: PDF, Markdown, HTML, and plain text documents
- **Semantic Chunking**: 200-500 token chunks with 15-20% overlap using RecursiveCharacterTextSplitter
- **OpenAI Embeddings**: Integration with `text-embedding-3-small` model
- **Vector Storage**: Qdrant collection with COSINE distance metric
- **Rich Metadata**: Document ID, chunk index, timestamps, character counts, and source tracking
- **CLI Interface**: Simple command-line tool for batch processing

## Files

- `ingest.py` - Main ingestion script with full pipeline implementation
- `test_basic.py` - Basic functionality tests (no API key required)
- All tests use OpenAI embeddings (test_without_openai.py removed)
- `verify_storage.py` - Verification and inspection tools
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Quick Start

### Prerequisites

1. **Qdrant running**: Start with `docker compose up -d qdrant` from `infrastructure/`
2. **Dependencies installed**: Run `pip install -r requirements.txt` in virtual environment
3. **OpenAI API Key**: Set `OPENAI_API_KEY` in `.env` file (for real usage)

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Test basic functionality (no API key needed)
python test_basic.py

# All tests use OpenAI embeddings

# Real ingestion (requires OpenAI API key)
python ingest.py --path ../../data

# Verify results
python verify_storage.py
```

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=http://localhost:6333
```

## Architecture

### Document Processing Pipeline

1. **Discovery**: Scan directory for supported file types
2. **Loading**: Use appropriate LangChain loader based on file extension
3. **Chunking**: Split into semantic chunks with RecursiveCharacterTextSplitter
4. **Embedding**: Generate vectors using OpenAI text-embedding-3-small
5. **Storage**: Store in Qdrant with comprehensive metadata

### Chunking Strategy

- **Chunk Size**: 500 characters (approximately 125-167 tokens)
- **Overlap**: 100 characters (20% overlap)
- **Separators**: `["\n\n", "\n", ". ", " ", ""]` for semantic boundaries
- **Index Tracking**: Maintains start positions for reconstruction

### Metadata Schema

Each stored chunk includes:

```json
{
  "doc_id": "path/to/document.md",
  "doc_name": "document.md", 
  "doc_type": ".md",
  "chunk_index": 0,
  "total_chunks": 6,
  "text": "Full chunk content...",
  "timestamp": "2025-08-11T17:21:19.995628",
  "char_count": 478,
  "start_index": 0
}
```

## Testing

The indexer includes comprehensive testing at multiple levels:

### Basic Tests (`test_basic.py`)
- Document loading verification
- Text splitting functionality
- Qdrant connection testing

### Integration Tests
- End-to-end pipeline with OpenAI embeddings
- Collection creation and data storage
- Metadata validation

### Verification (`verify_storage.py`)
- Collection inspection and statistics
- Chunk content verification
- Basic search functionality testing

## Performance

- **Chunking**: ~6 chunks for 2KB sample document
- **Storage**: 1536-dimensional vectors with COSINE distance
- **Metadata**: Rich payload with full text preservation
- **Scalability**: Supports batch processing of document directories

## Phase 1 Acceptance Criteria ✅

All acceptance criteria from PLAN.md have been met:

- ✅ Qdrant collection exists with expected vector count
- ✅ Spot-check stored chunks for correct text & metadata
- ✅ Sample corpus indexed with coherent metadata
- ✅ CLI interface `python ingest.py --path ./data`

## Next Steps

Phase 1 is complete and ready for Phase 2: Retrieval Core (Hybrid + Rerank)

- Implement hybrid retrieval (dense + BM25)
- Add reranking with cross-encoder models
- Implement MMR for diversity
- Create retrieval API endpoints
