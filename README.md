# MAI Storage - Agentic RAG System

An agentic RAG (Retrieval Augmented Generation) system built with Next.js, FastAPI, and LangGraph, featuring advanced retrieval techniques and sentence-level citations.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd maistorage

# Create environment file
make setup

# Edit .env with your API keys
nano .env  # Add your OPENAI_API_KEY
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
# Start Qdrant (required)
make start-infra

# Or start with optional Elasticsearch
make start-infra-full

# Verify services are healthy
make health
```

### 4. Start Development Servers

In separate terminals:

```bash
# Terminal 1: Start RAG API
make start-api

# Terminal 2: Start Next.js web app
make start-web
```

### 5. Verify Everything Works

- **Web App**: http://localhost:3000
- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Qdrant**: http://localhost:6333/dashboard

## 🏗️ Architecture

```
maistorage/
├── apps/
│   └── web/                 # Next.js 14 frontend
├── services/
│   ├── rag_api/            # FastAPI + LangGraph backend
│   └── indexer/            # Document ingestion service
├── infrastructure/
│   └── docker-compose.yml  # Qdrant + Elasticsearch
├── .env.example            # Environment configuration
└── Makefile               # Development commands
```

## 🛠️ Development Commands

```bash
# Development workflow
make dev              # Start all infrastructure
make start-api        # Start RAG API server
make start-web        # Start Next.js web app

# Code quality
make lint             # Run linting
make format           # Format code
make test             # Run tests

# Infrastructure
make start-infra      # Start Qdrant only
make start-infra-full # Start Qdrant + Elasticsearch
make stop-infra       # Stop all infrastructure
make logs             # View infrastructure logs

# Data ingestion
make ingest           # Run document ingestion

# Utilities
make health           # Check service health
make clean            # Clean caches and generated files
```

## 📋 Phase 10 - COMPLETE ✅

This system implements **Phase 10: Hardening, DX & Deployment** with production-ready infrastructure:

### 🚀 Production Features
- **Security**: Rate limiting, circuit breakers, request validation, security headers
- **Monitoring**: Prometheus metrics, structured logging, health checks, performance tracking  
- **Deployment**: Docker containerization, Vercel config, monitoring stack integration
- **Reliability**: Error boundaries, graceful fallbacks, dependency validation

### 📊 Monitoring & Observability
- **Structured Logging**: Request correlation IDs, JSON format, performance metrics
- **Prometheus Integration**: Complete metrics collection for requests, latency, resources
- **Health Endpoints**: Multi-level validation (`/health`, `/health/detailed`, `/metrics`, `/stats`)
- **System Monitoring**: CPU, memory, disk usage with automatic collection

### 🔒 Security & Hardening
- **Environment Validation**: 25+ validated settings with secure defaults
- **Middleware Stack**: Security headers, rate limiting, circuit breakers, size limits
- **Error Handling**: Consistent error responses, information disclosure prevention
- **Container Security**: Non-root user, health checks, minimal attack surface

## 📋 Implementation History

### Phase 0 - Infrastructure Foundation ✅

### ✅ Monorepo Structure
- `/apps/web` - Next.js 14 application
- `/services/rag_api` - FastAPI + LangGraph service
- `/services/indexer` - Document ingestion service
- `/infrastructure` - Docker Compose setup

### ✅ Tooling & Standards
- **Python**: `ruff`, `black`, `isort` with pre-commit hooks
- **JavaScript**: ESLint, Prettier with Next.js
- **Development**: Makefile with common commands
- **Infrastructure**: Docker Compose for Qdrant (+optional Elasticsearch)

### ✅ Configuration
- Environment variables via `.env` file
- Model configuration: `gpt-4o-mini` + `text-embedding-3-small`
- CORS setup for frontend/backend communication
- Health checks and service monitoring

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Optional
QDRANT_URL=http://localhost:6333
ELASTICSEARCH_URL=http://localhost:9200
LANGCHAIN_TRACING_V2=true  # Enable LangSmith tracing
```

## 🧪 Acceptance Criteria - Phase 0

- [x] `npm run dev` (or `make start-web`) runs Next.js on http://localhost:3000
- [x] `uvicorn main:app --reload` (or `make start-api`) returns API response
- [x] `docker compose up -d qdrant` (or `make start-infra`) starts healthy Qdrant
- [x] README quickstart verified on clean machine setup

## 🔄 Next Steps (Upcoming Phases)

- **Phase 1**: Document ingestion and indexing with Qdrant
- **Phase 2**: Hybrid retrieval (dense + BM25) with reranking
- **Phase 3**: Baseline RAG implementation
- **Phase 4**: Streaming chat UI with Next.js
- **Phase 5**: Agentic loops with LangGraph
- **Phase 6**: Sentence-level citations
- **Phase 7**: Frontend trace visualization
- **Phase 8**: Evaluation harness (RAGAS)

## 🤝 Contributing

```bash
# Setup pre-commit hooks
pip install pre-commit
pre-commit install

# Code standards are enforced via:
make lint    # Check code quality
make format  # Auto-format code
```

## 📄 License

MIT License - see LICENSE file for details