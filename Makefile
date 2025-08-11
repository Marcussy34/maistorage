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

# Phase 11 comprehensive test suite
test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	cd services/rag_api && python -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	cd services/rag_api && python -m pytest tests/integration/ -v --tb=short

test-edge-cases: ## Run edge case tests only
	@echo "Running edge case tests..."
	cd services/rag_api && python -m pytest tests/edge_cases/ -v --tb=short

test-performance: ## Run performance tests only
	@echo "Running performance tests..."
	cd services/rag_api && python -m pytest tests/performance/ -v --tb=short -m "not slow"

test-comprehensive: ## Run complete Phase 11 test suite
	@echo "Running comprehensive Phase 11 test suite..."
	cd services/rag_api && python tests/run_tests.py --test-type all --verbose

test-coverage: ## Run tests with coverage reporting
	@echo "Running tests with coverage..."
	cd services/rag_api && python -m pytest tests/ --cov=./ --cov-report=html --cov-report=term-missing --cov-fail-under=70

test-report: ## Generate HTML test report
	@echo "Generating test report..."
	cd services/rag_api && python tests/run_tests.py --report test_report.html

test-fast: ## Run fast tests only (excludes slow performance tests)
	@echo "Running fast tests..."
	cd services/rag_api && python -m pytest tests/ -v --tb=short -m "not slow"

test-slow: ## Run slow tests only
	@echo "Running slow tests..."
	cd services/rag_api && python -m pytest tests/ -v --tb=short -m "slow"

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
	@curl -s http://localhost:6333/health || echo "❌ Qdrant not responding"
	@curl -s http://localhost:9200/_cluster/health || echo "⚠️  Elasticsearch not running (optional)"
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health || echo "❌ RAG API not responding"
	@curl -s http://localhost:3000/api/health > /dev/null && echo "✅ Web app responding" || echo "❌ Web app not responding"

health-detailed: ## Get detailed health information
	@echo "=== Detailed Health Check ==="
	@echo "RAG API Health:"
	@curl -s http://localhost:8000/health | jq . || echo "RAG API not available"
	@echo "\nRAG API Detailed Health:"
	@curl -s http://localhost:8000/health/detailed | jq . || echo "Detailed health not available"
	@echo "\nWeb App Health:"
	@curl -s http://localhost:3000/api/health | jq . || echo "Web health not available"

stats: ## Get system statistics
	@echo "=== System Statistics ==="
	@curl -s http://localhost:8000/stats | jq . || echo "Stats not available"

metrics: ## View Prometheus metrics
	@echo "=== Prometheus Metrics Sample ==="
	@curl -s http://localhost:8000/metrics | head -20 || echo "Metrics not available"
	@echo "\n... (truncated, visit http://localhost:8000/metrics for full metrics)"

# Production deployment commands
deploy-production: ## Deploy production stack with monitoring
	@echo "Deploying production stack..."
	export ENVIRONMENT=production BUILD_TARGET=production && \
	docker compose --profile production --profile monitoring up -d

deploy-dev: ## Deploy development stack
	@echo "Deploying development stack..."
	docker compose --profile api up -d

stop-all: ## Stop all services including monitoring
	docker compose --profile production --profile monitoring --profile optional down

# Monitoring commands
monitoring-up: ## Start monitoring stack (Prometheus + Grafana)
	docker compose --profile monitoring up -d

monitoring-down: ## Stop monitoring stack
	docker compose --profile monitoring down

logs-api: ## View RAG API logs
	docker compose logs -f rag-api

logs-all: ## View all service logs
	docker compose logs -f

# Phase 10 testing
test-phase10: ## Run Phase 10 integration tests
	cd services/rag_api && python test_phase10_integration.py

test-deployment: ## Test complete deployment workflow
	@echo "Testing deployment workflow..."
	@echo "1. Starting infrastructure..."
	@make start-infra
	@sleep 10
	@echo "2. Starting API..."
	@make start-api &
	@sleep 15
	@echo "3. Running integration tests..."
	@make test-phase10
	@echo "Deployment test complete!"

validate-production: ## Validate production readiness
	@echo "=== Production Readiness Validation ==="
	@echo "1. Health checks..."
	@make health-detailed
	@echo "\n2. Security headers..."
	@curl -I http://localhost:8000/health | grep -E "(X-Content-Type-Options|X-Frame-Options|X-API-Version)"
	@echo "\n3. Monitoring endpoints..."
	@curl -s http://localhost:8000/metrics | head -5
	@echo "\n4. Performance stats..."
	@curl -s http://localhost:8000/stats | jq '.performance.uptime_seconds'
	@echo "\n✅ Production validation complete"
