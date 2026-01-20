.PHONY: install dev test lint format typecheck docs serve clean all

# Default target
all: install lint typecheck test

# Install dependencies
install:
	poetry install

# Install with dev dependencies
dev:
	poetry install --with dev,test

# Run tests
test:
	poetry run pytest tests/ -v --cov=contextflow --cov-report=term-missing

# Run tests with coverage report
test-cov:
	poetry run pytest tests/ -v --cov=contextflow --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Run linting
lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

# Format code
format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

# Type checking
typecheck:
	poetry run mypy src/contextflow/

# Build documentation
docs:
	poetry run mkdocs build

# Serve documentation locally
docs-serve:
	poetry run mkdocs serve

# Start API server
serve:
	poetry run uvicorn contextflow.api.server:app --reload --port 8000

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build:
	poetry build

# Publish to PyPI (use with caution)
publish:
	poetry publish --build

# Create new release
release:
	@echo "Current version: $$(poetry version -s)"
	@read -p "New version: " version; \
	poetry version $$version; \
	git add pyproject.toml; \
	git commit -m "Bump version to $$version"; \
	git tag -a "v$$version" -m "Release v$$version"

# Run pre-commit hooks
pre-commit:
	poetry run pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	poetry run pre-commit install

# Quick check (fast feedback loop)
check: format lint typecheck
	@echo "All checks passed!"
