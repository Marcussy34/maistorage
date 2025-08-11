.PHONY: help install-deps start-infra stop-infra start-api start-web dev lint format clean test

# Default target
help: ## Show this help message
	@echo "MAI Storage Development Commands"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install-deps: ## Install all dependencies
	@echo "Installing Python dependencies for RAG API..."
	cd services/rag_api && source venv/bin/activate && pip install -r requirements.txt
	@echo "Installing Python dependencies for Indexer..."
	cd services/indexer && source venv/bin/activate && pip install -r requirements.txt
	@echo "Installing Node.js dependencies for Web..."
	cd apps/web && npm install

setup: ## Initial setup (copy .env.example to .env)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please edit .env with your API keys"; \
	else \
		echo ".env file already exists"; \
	fi

# Infrastructure
start-infra: ## Start infrastructure services (Qdrant, optional Elasticsearch)
	docker compose -f infrastructure/docker-compose.yml up -d qdrant

start-infra-full: ## Start all infrastructure services including Elasticsearch
	docker compose -f infrastructure/docker-compose.yml --profile optional up -d

stop-infra: ## Stop infrastructure services
	docker compose -f infrastructure/docker-compose.yml down

restart-infra: stop-infra start-infra ## Restart infrastructure services

# Services
start-api: ## Start the RAG API server
	cd services/rag_api && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

start-web: ## Start the Next.js web application
	cd apps/web && npm run dev

# Development
dev: ## Start all services for development
	@echo "Starting infrastructure..."
	$(MAKE) start-infra
	@echo "Waiting for services to be ready..."
	sleep 5
	@echo "Infrastructure ready! Now start the API and Web servers in separate terminals:"
	@echo "  make start-api"
	@echo "  make start-web"

# Code quality
lint: ## Run linting on all code
	@echo "Linting Python code..."
	cd services/rag_api && ruff check . && black --check .
	cd services/indexer && ruff check . && black --check .
	@echo "Linting JavaScript/TypeScript code..."
	cd apps/web && npm run lint

format: ## Format all code
	@echo "Formatting Python code..."
	cd services/rag_api && black . && isort .
	cd services/indexer && black . && isort .
	@echo "Formatting completed"

# Testing and utilities
test: ## Run all tests
	@echo "Running Python tests..."
	cd services/rag_api && python -m pytest tests/ || echo "No tests yet"
	cd services/indexer && python -m pytest tests/ || echo "No tests yet"
	@echo "Running JavaScript tests..."
	cd apps/web && npm test || echo "No tests configured yet"

ingest: ## Run document ingestion (example)
	cd services/indexer && python ingest.py --path ./data

clean: ## Clean up generated files and caches
	@echo "Cleaning Python caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaning Node.js caches..."
	cd apps/web && rm -rf .next node_modules/.cache 2>/dev/null || true
	@echo "Cleanup completed"

logs: ## Show infrastructure logs
	docker compose -f infrastructure/docker-compose.yml logs -f

health: ## Check health of all services
	@echo "Checking infrastructure health..."
	@curl -s http://localhost:6333/health || echo "Qdrant not responding"
	@curl -s http://localhost:9200/_cluster/health || echo "Elasticsearch not running (optional)"
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health || echo "RAG API not responding"
	@echo "Checking Web..."
	@curl -s http://localhost:3000 > /dev/null && echo "Web app responding" || echo "Web app not responding"
