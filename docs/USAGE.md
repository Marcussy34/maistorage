# MAI Storage - Usage Guide

**Version**: Phase 12 Complete  
**Last Updated**: January 2025  
**For**: Developers, System Administrators, and End Users

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Configuration](#environment-configuration)
3. [Development Workflow](#development-workflow)
4. [API Reference](#api-reference)
5. [Web Interface Guide](#web-interface-guide)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Prerequisites
- **Python 3.11+** with pip or uv
- **Node.js 18+** with npm or pnpm
- **Docker & Docker Compose** for infrastructure
- **OpenAI API Key** for LLM access

### 1. Repository Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd maistorage

# Create environment configuration
make setup

# Edit .env with your API keys
nano .env  # Add OPENAI_API_KEY=sk-your-key-here
```

### 2. Install Dependencies
```bash
# Install all dependencies (Python + Node.js)
make install-deps

# Or install manually:
cd services/rag_api && pip install -r requirements.txt
cd ../../services/indexer && pip install -r requirements.txt  
cd ../../apps/web && npm install
```

### 3. Start Infrastructure
```bash
# Start Qdrant vector database (required)
make start-infra

# Or start with optional Elasticsearch for hybrid search
make start-infra-full

# Verify services are healthy
make health
```

### 4. Run the Application
```bash
# Terminal 1: Start RAG API server
make start-api

# Terminal 2: Start Next.js web application
make start-web

# Or run everything in background
make dev
```

### 5. Access the Application
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

---

## Environment Configuration

### Essential Environment Variables

Create a `.env` file in the project root:

```bash
# Required: OpenAI API Configuration
OPENAI_API_KEY=***REMOVED***
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for proxies

# Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                             # Optional: for production
ELASTICSEARCH_URL=http://localhost:9200     # Optional: for hybrid search

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
CORS_ORIGINS=["http://localhost:3000"]

# Performance & Caching
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
MAX_QUERY_LENGTH=1000
MAX_CONTEXT_LENGTH=8000

# Monitoring & Logging (Optional)
LOG_LEVEL=INFO
ENABLE_METRICS=true
LANGCHAIN_TRACING_V2=true                   # Enable LangSmith tracing
LANGCHAIN_API_KEY=your-langsmith-key        # Optional: for LangSmith
```

### Configuration Validation

The system validates all environment variables on startup:

```bash
# Check configuration validity
make validate-config

# View current configuration
make show-config
```

---

## Development Workflow

### Daily Development Commands

```bash
# Start everything for development
make dev                    # Full stack with hot reload

# Individual service control
make start-infra           # Qdrant only
make start-infra-full      # Qdrant + Elasticsearch
make start-api             # FastAPI server
make start-web             # Next.js development server

# Stop services
make stop-infra            # Stop infrastructure
make stop                  # Stop all services

# View logs
make logs                  # Infrastructure logs
make logs-api              # API server logs
```

### Code Quality & Testing

```bash
# Code formatting and linting
make format                # Auto-format Python + JavaScript
make lint                  # Check code quality
make lint-fix              # Auto-fix linting issues

# Testing
make test                  # Run all tests
make test-unit             # Unit tests only (fast)
make test-integration      # Integration tests
make test-comprehensive    # Full test suite with reporting

# Coverage analysis
make test-coverage         # Generate HTML coverage report
```

### Data Management

```bash
# Document ingestion
make ingest                # Ingest documents from /data
make ingest-sample         # Ingest sample document only

# Data verification
make verify-storage        # Check Qdrant collections
make verify-indexer        # Validate ingestion pipeline

# Data cleanup
make clean-data            # Clear Qdrant collections
make clean-cache           # Clear application caches
```

---

## API Reference

### Core Endpoints

#### Health & Status
```bash
# Basic health check
GET /health
# Response: {"status": "healthy", "timestamp": "2025-01-27T..."}

# Detailed health with dependencies
GET /health/detailed
# Response: {"status": "healthy", "services": {...}, "metrics": {...}}

# Prometheus metrics
GET /metrics
# Response: Prometheus-format metrics

# System statistics
GET /stats
# Response: {"uptime": 3600, "requests": 150, "memory_mb": 512}
```

#### Document Retrieval
```bash
# Search documents
POST /retrieve
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 10,
  "retrieval_method": "hybrid",
  "rerank_method": "bge",
  "enable_mmr": true,
  "mmr_diversity_threshold": 0.3
}

# Response:
{
  "query": "What is machine learning?",
  "results": [
    {
      "document": {
        "id": "doc_123",
        "content": "Machine learning is...",
        "metadata": {
          "title": "Introduction to ML",
          "source": "ml_guide.pdf",
          "page": 1
        }
      },
      "score": 0.95,
      "retrieval_method": "hybrid"
    }
  ],
  "total_results": 5,
  "retrieval_time_ms": 250
}
```

#### Traditional RAG
```bash
# Single-pass RAG
POST /rag
Content-Type: application/json

{
  "query": "Explain transformer architecture",
  "top_k": 5,
  "enable_sentence_citations": true
}

# Response:
{
  "query": "Explain transformer architecture",
  "answer": "Transformers are neural network architectures...",
  "citations": [
    {
      "source_id": "doc_456",
      "content": "The transformer architecture introduced...",
      "score": 0.89,
      "title": "Attention Is All You Need"
    }
  ],
  "response_time_ms": 1200,
  "token_usage": {
    "prompt_tokens": 850,
    "completion_tokens": 300,
    "total_tokens": 1150
  }
}
```

#### Agentic RAG with Streaming
```bash
# Multi-step agentic workflow with real-time updates
POST /chat/stream?agentic=true
Content-Type: application/json

{
  "query": "Compare machine learning and deep learning approaches",
  "top_k": 10,
  "max_refinements": 2,
  "enable_verification": true,
  "stream_trace": true
}

# Response: NDJSON stream of events
{"type": "step_start", "step": "planner", "timestamp": "2025-01-27T10:00:00Z"}
{"type": "step_complete", "step": "planner", "data": {"sub_queries": ["machine learning definition", "deep learning definition"], "key_concepts": ["algorithms", "neural networks"]}}
{"type": "step_start", "step": "retriever", "timestamp": "2025-01-27T10:00:01Z"}
{"type": "sources", "data": {"results": [...], "retrieval_time_ms": 300}}
{"type": "step_start", "step": "synthesizer", "timestamp": "2025-01-27T10:00:02Z"}
{"type": "token", "data": {"token": "Machine", "position": 0}}
{"type": "token", "data": {"token": " learning", "position": 1}}
{"type": "step_complete", "step": "synthesizer", "data": {"answer": "Machine learning and deep learning..."}}
{"type": "step_start", "step": "verifier", "timestamp": "2025-01-27T10:00:05Z"}
{"type": "verification", "data": {"needs_refinement": false, "confidence": 0.92, "coverage_score": 0.88}}
{"type": "metrics", "data": {"total_time_ms": 5200, "tokens_used": 1850, "refinement_count": 0}}
{"type": "done", "data": {"final_answer": "...", "citations": [...], "trace_summary": {...}}}
```

### Request/Response Models

#### RetrievalRequest
```python
{
  "query": str,                    # User query (required)
  "top_k": int = 10,              # Number of results to return
  "retrieval_method": str = "hybrid",  # "dense", "sparse", "hybrid"
  "rerank_method": str = "bge",   # "none", "bge"
  "enable_mmr": bool = True,      # Enable diversity via MMR
  "mmr_diversity_threshold": float = 0.3,
  "include_metadata": bool = True
}
```

#### RAGRequest  
```python
{
  "query": str,                    # User query (required)
  "top_k": int = 5,               # Context documents
  "enable_sentence_citations": bool = False,
  "temperature": float = 0.7,     # LLM creativity
  "max_tokens": int = 1000        # Response length limit
}
```

#### ChatStreamRequest
```python
{
  "query": str,                    # User query (required)
  "top_k": int = 10,              # Retrieval depth
  "max_refinements": int = 2,     # Agentic refinement loops
  "enable_verification": bool = True,
  "stream_trace": bool = True,    # Include workflow events
  "stream_tokens": bool = False   # Stream individual tokens (future)
}
```

---

## Web Interface Guide

### Chat Interface (`/chat`)

#### Basic Usage
1. **Open**: Navigate to http://localhost:3000/chat
2. **Mode Selection**: Toggle between "Traditional" and "Agentic" RAG
3. **Query Input**: Type your question in the input field
4. **Send**: Press Enter or click Send button
5. **View Response**: Watch streaming response with citations

#### Features Overview

##### Mode Toggle
- **Traditional RAG**: Single-pass retrieval and generation
- **Agentic RAG**: Multi-step reasoning with planner, retriever, synthesizer, verifier

##### Agent Trace Panel (Agentic Mode)
- **Timeline View**: Step-by-step execution progress
- **Performance Metrics**: Timing and token usage for each step
- **Verification Results**: Quality scores and refinement decisions
- **Sub-query Tracking**: See how complex queries are decomposed

##### Citation System
- **Inline Citations**: Numbered superscript references (¹ ² ³)
- **Hover Cards**: Preview source content on citation hover
- **Source Panel**: Full document context in side panel
- **Confidence Indicators**: ⚠️ for low-confidence claims

##### Context Panel
- **Source Documents**: Top-k retrieved documents
- **Relevance Scores**: Hybrid search and reranking scores
- **Metadata Display**: Document titles, pages, sections
- **Highlight Matching**: Query terms highlighted in context

#### Keyboard Shortcuts
- **Enter**: Send message
- **Shift + Enter**: New line in input
- **Ctrl/Cmd + K**: Clear conversation
- **Ctrl/Cmd + /**: Toggle mode (Traditional ↔ Agentic)

### Evaluation Dashboard (`/eval`)

#### Metrics Overview
- **RAGAS Scores**: Faithfulness, Answer Relevancy, Context Precision/Recall
- **Retrieval Metrics**: Recall@k, nDCG, MRR
- **Performance Data**: Latency distributions, token usage
- **Comparison Charts**: Traditional vs Agentic side-by-side

#### Running Evaluations
1. **Navigate**: Go to http://localhost:3000/eval
2. **Select Mode**: Choose Traditional, Agentic, or Both
3. **Configure**: Set evaluation parameters (top_k, dataset)
4. **Run**: Click "Start Evaluation" button
5. **Monitor**: Watch real-time progress and results
6. **Export**: Download results as JSON or CSV

---

## Configuration Options

### Retrieval Configuration

#### Qdrant Settings
```yaml
# Vector database configuration
qdrant:
  url: "http://localhost:6333"
  collection_name: "mai_storage_vectors"
  vector_size: 1536
  distance: "Cosine"
  
  # HNSW parameters for performance tuning
  hnsw_config:
    m: 48                    # Graph connectivity (16-64)
    ef_construction: 256     # Index building accuracy (100-500)
    ef_search: 128          # Query time accuracy (50-200)
```

#### Embedding Configuration
```yaml
# Text embedding settings
embeddings:
  model: "text-embedding-3-small"
  dimensions: 1536
  batch_size: 100
  normalize: true
  cache_ttl_hours: 24
```

#### Hybrid Search Configuration
```yaml
# Search algorithm settings
retrieval:
  dense_weight: 0.6        # Dense search importance (0.0-1.0)
  sparse_weight: 0.4       # BM25 search importance (0.0-1.0)
  fusion_method: "rrf"     # "rrf" or "weighted"
  rrf_k: 60               # RRF parameter (higher = more diverse)
  
  # Reranking settings
  rerank_top_k: 100       # Candidates to rerank
  rerank_final_k: 15      # Final results after reranking
  
  # MMR diversity settings
  mmr_diversity_threshold: 0.3  # Higher = more diverse (0.0-1.0)
  mmr_top_k: 50          # Candidates for MMR selection
```

### LLM Configuration

#### OpenAI Settings
```yaml
# Language model configuration
llm:
  model: "gpt-4o-mini"
  temperature: 0.7         # Creativity (0.0-2.0)
  max_tokens: 4096        # Response length limit
  timeout_seconds: 30     # Request timeout
  retry_attempts: 3       # Failure retry count
  retry_delay: 1.0       # Delay between retries (seconds)
```

#### Prompt Templates
```yaml
# Customizable prompt templates
prompts:
  baseline_system: |
    You are a helpful AI assistant specializing in information retrieval.
    Provide accurate, well-cited responses based on the given context.
    
  planner_system: |
    You are a query planning specialist. Break down complex questions
    into sub-queries and identify key concepts for effective retrieval.
    
  verifier_system: |
    You are a quality assurance specialist. Evaluate answer quality,
    faithfulness to sources, and completeness of coverage.
```

### Agentic Workflow Configuration

#### LangGraph Settings
```yaml
# Agentic workflow parameters
agentic:
  max_refinements: 2       # Maximum refinement loops
  enable_verification: true
  verification_threshold: 0.8  # Quality threshold for approval
  
  # Step timeouts (seconds)
  planner_timeout: 15
  retriever_timeout: 30
  synthesizer_timeout: 45
  verifier_timeout: 20
  
  # Planning parameters
  max_sub_queries: 5       # Maximum query decomposition
  min_key_concepts: 2      # Minimum concepts to extract
```

#### Citation Engine Configuration
```yaml
# Sentence-level citation settings
citations:
  enable_sentence_level: true
  confidence_threshold: 0.7    # Minimum confidence for citation
  max_citations_per_sentence: 3
  attribution_method: "embedding"  # "embedding" or "llm"
  
  # Warning thresholds
  low_confidence_warning: true
  warning_threshold: 0.5       # Show ⚠️ below this confidence
```

### Performance Configuration

#### Caching Settings
```yaml
# Multi-layer caching configuration
cache:
  enabled: true
  
  # Cache layer TTLs (hours)
  embedding_ttl: 24
  candidate_ttl: 1
  reranker_ttl: 6
  llm_response_ttl: 24
  prompt_template_ttl: 168   # 7 days
  
  # Cache sizes (number of items)
  max_embedding_cache: 10000
  max_candidate_cache: 1000
  max_reranker_cache: 5000
  max_llm_cache: 2000
```

#### Concurrency Settings
```yaml
# Async processing configuration
concurrency:
  max_concurrent_requests: 50
  max_concurrent_retrievals: 10
  max_concurrent_llm_calls: 5
  
  # Resource limits
  max_memory_mb: 2048
  max_cpu_percent: 80
  request_timeout_seconds: 120
```

---

## Troubleshooting

### Common Issues

#### 1. API Connection Issues
```bash
# Check API health
curl http://localhost:8000/health

# Verify environment variables
make show-config

# Check API logs
make logs-api

# Common fixes:
export OPENAI_API_KEY=sk-your-key-here
make restart-api
```

#### 2. Qdrant Connection Problems
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Restart Qdrant
make stop-infra
make start-infra

# Check collections
curl http://localhost:6333/collections
```

#### 3. Slow Response Times
```bash
# Check current performance
curl http://localhost:8000/stats

# Optimize Qdrant HNSW parameters
# Edit .env: QDRANT_EF_SEARCH=256 (higher = slower but more accurate)

# Enable caching
# Edit .env: ENABLE_CACHING=true

# Check memory usage
make monitor
```

#### 4. Poor Search Quality
```bash
# Run evaluation to measure quality
make eval-run

# Check retrieval metrics
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 10}'

# Tune search parameters:
# - Increase top_k for more context
# - Adjust dense_weight vs sparse_weight
# - Enable MMR for diversity
```

#### 5. Empty or Incorrect Results
```bash
# Verify data ingestion
make verify-storage

# Re-ingest documents
make clean-data
make ingest

# Check embedding model
# Ensure EMBEDDING_MODEL matches indexed embeddings
```

### Error Codes & Solutions

#### HTTP 422 Validation Error
```json
{"detail": [{"loc": ["body", "query"], "msg": "field required", "type": "value_error.missing"}]}
```
**Solution**: Ensure request includes required fields (query, etc.)

#### HTTP 500 Internal Server Error
```json
{"error": "OpenAI API request failed", "detail": "Incorrect API key provided"}
```
**Solution**: Verify OPENAI_API_KEY in .env file

#### HTTP 503 Service Unavailable
```json
{"error": "Qdrant connection failed", "detail": "Connection refused"}
```
**Solution**: Start Qdrant with `make start-infra`

### Debug Mode

#### Enable Verbose Logging
```bash
# Edit .env
LOG_LEVEL=DEBUG
ENABLE_TRACE_LOGGING=true

# Restart API
make restart-api

# View detailed logs
make logs-api
```

#### Performance Profiling
```bash
# Enable performance monitoring
export ENABLE_METRICS=true
export ENABLE_PROFILING=true

# View metrics
curl http://localhost:8000/metrics

# Generate performance report
make profile-report
```

---

## Advanced Usage

### Custom Document Ingestion

#### Supported Formats
- **PDF**: Text extraction with metadata
- **Markdown**: Native parsing with sections
- **HTML**: Content extraction with structure
- **Plain Text**: Direct processing
- **JSON**: Structured document format

#### Custom Ingestion Script
```python
#!/usr/bin/env python3
"""Custom document ingestion example."""

from services.indexer.ingest import DocumentIngester

# Initialize ingester
ingester = DocumentIngester(
    qdrant_url="http://localhost:6333",
    collection_name="custom_collection",
    embedding_model="text-embedding-3-small"
)

# Ingest custom documents
documents = [
    {
        "content": "Your document content here...",
        "metadata": {
            "title": "Custom Document",
            "source": "custom_source.pdf",
            "author": "Your Name",
            "tags": ["ai", "rag", "custom"]
        }
    }
]

# Process and index
results = await ingester.ingest_documents(documents)
print(f"Indexed {len(results)} documents")
```

### Custom Prompt Templates

#### Baseline RAG Customization
```python
# File: services/rag_api/prompts/custom_baseline.py

CUSTOM_BASELINE_PROMPT = """
You are a specialized AI assistant for {domain}.

Context Information:
{context}

User Question: {query}

Instructions:
1. Provide accurate, detailed answers based solely on the context
2. Use technical terminology appropriate for {domain}
3. Include specific citations using [source_id] format
4. If information is insufficient, clearly state limitations

Answer:
"""

def format_custom_baseline_prompt(query: str, context: str, domain: str = "general") -> str:
    return CUSTOM_BASELINE_PROMPT.format(
        query=query,
        context=context,
        domain=domain
    )
```

### Advanced Retrieval Configuration

#### Custom Similarity Functions
```python
# File: services/rag_api/custom_retrieval.py

from typing import List, Tuple
import numpy as np

def custom_similarity_score(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    metadata_boost: List[float]
) -> np.ndarray:
    """Custom similarity with metadata boosting."""
    
    # Base cosine similarity
    base_scores = np.dot(query_embedding, doc_embeddings.T)
    
    # Apply metadata boost (e.g., recency, authority)
    boosted_scores = base_scores * np.array(metadata_boost)
    
    return boosted_scores

def domain_specific_reranking(
    candidates: List[dict],
    query: str,
    domain_keywords: List[str]
) -> List[dict]:
    """Rerank based on domain-specific relevance."""
    
    for candidate in candidates:
        content = candidate["content"].lower()
        
        # Boost score for domain keywords
        keyword_score = sum(1 for kw in domain_keywords if kw in content)
        candidate["score"] *= (1 + 0.1 * keyword_score)
    
    return sorted(candidates, key=lambda x: x["score"], reverse=True)
```

### Production Deployment

#### Docker Production Setup
```dockerfile
# File: services/rag_api/Dockerfile.prod

FROM python:3.11-slim

# Production optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHON_ENV=production

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120"]
```

#### Kubernetes Deployment
```yaml
# File: infrastructure/k8s/rag-api-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: maistorage/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Monitoring & Analytics

#### Custom Metrics Collection
```python
# File: services/rag_api/custom_metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Custom business metrics
QUERY_COMPLEXITY = Histogram(
    'query_complexity_score',
    'Distribution of query complexity scores',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

CITATION_ACCURACY = Gauge(
    'citation_accuracy_score',
    'Current citation accuracy score'
)

AGENTIC_REFINEMENTS = Counter(
    'agentic_refinement_total',
    'Total number of agentic refinements performed'
)

def track_query_metrics(query: str, response_data: dict):
    """Track custom query metrics."""
    
    # Measure query complexity (word count, special terms, etc.)
    complexity = calculate_query_complexity(query)
    QUERY_COMPLEXITY.observe(complexity)
    
    # Track citation accuracy if available
    if 'citation_scores' in response_data:
        avg_citation_score = np.mean(response_data['citation_scores'])
        CITATION_ACCURACY.set(avg_citation_score)
    
    # Count refinements in agentic mode
    if response_data.get('refinement_count', 0) > 0:
        AGENTIC_REFINEMENTS.inc(response_data['refinement_count'])
```

---

This usage guide provides comprehensive coverage of all system capabilities, from basic setup to advanced customization and production deployment. For additional support, refer to the troubleshooting section or check the project's GitHub issues.
