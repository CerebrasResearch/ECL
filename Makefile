.PHONY: help env install dev test lint format typecheck clean figures tables all-experiments

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda environment
	bash create_env.sh

install:  ## Install package in editable mode
	pip install -e .

dev:  ## Install with dev + model dependencies
	pip install -e ".[all]"
	pre-commit install

test:  ## Run test suite
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest tests/ -v --tb=short --cov=ecl --cov-report=term-missing --cov-report=html

lint:  ## Run linter
	ruff check src/ tests/ scripts/

format:  ## Auto-format code
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck:  ## Run type checker
	mypy src/ecl/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

figures:  ## Generate all figures
	@for f in scripts/fig*.py; do echo "=== Running $$f ==="; python "$$f"; done

tables:  ## Generate all tables
	@for f in scripts/tab*.py; do echo "=== Running $$f ==="; python "$$f"; done

all-experiments: figures tables  ## Run all experiment scripts
