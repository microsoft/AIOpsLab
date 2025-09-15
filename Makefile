# Makefile for AIOpsLab Task Execution API

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Database
.PHONY: db-up
db-up: ## Start PostgreSQL with Docker Compose
	docker-compose up -d postgres

.PHONY: db-down
db-down: ## Stop PostgreSQL
	docker-compose down

.PHONY: db-reset
db-reset: ## Reset database (drop and recreate)
	docker-compose down -v
	docker-compose up -d postgres
	sleep 3

# Testing
.PHONY: test
test: ## Run quick tests
	cd task-executor/api && PYTHONPATH=. TESTING=true CREATE_ENGINE=false pytest tests/ -v --tb=short

.PHONY: test-all
test-all: ## Run all tests with full output
	cd task-executor/api && PYTHONPATH=. TESTING=true CREATE_ENGINE=false pytest tests/ -v

# API Server
.PHONY: api-dev
api-dev: ## Run API server in development mode (with integrated workers)
	cd task-executor/api && PYTHONPATH=. uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: api-run
api-run: ## Run API server in production mode
	cd task-executor/api && PYTHONPATH=. uvicorn src.main:app --host 0.0.0.0 --port 8000

# Docker
.PHONY: docker-build
docker-build: ## Build Docker images
	docker-compose build

.PHONY: docker-up
docker-up: ## Start all services with Docker Compose
	docker-compose up -d

.PHONY: docker-down
docker-down: ## Stop all services
	docker-compose down

.PHONY: docker-logs
docker-logs: ## Show logs from all services
	docker-compose logs -f

# Code Quality
.PHONY: format
format: ## Format code with black
	black task-executor/api/src task-executor/api/tests

.PHONY: lint
lint: ## Run linting with ruff
	ruff check task-executor/api/src task-executor/api/tests

.PHONY: clean
clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Development shortcuts
.PHONY: dev
dev: db-up ## Start development environment
	@echo "Starting development environment..."
	@echo "Database is running at localhost:5432"
	@echo "Run 'make api-dev' to start the API server (workers start automatically)"

.PHONY: stop
stop: docker-down ## Stop all services

.PHONY: restart
restart: stop dev ## Restart all services