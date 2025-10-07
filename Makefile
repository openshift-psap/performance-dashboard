# Makefile for Performance Dashboard
# Provides convenient commands for development and CI/CD

.PHONY: help install install-dev lint format type-check security test clean pre-commit setup-dev

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-dev     Set up development environment"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  lint          Run all linting checks"
	@echo "  format        Format code with ruff"
	@echo "  format-check  Check code formatting without making changes"
	@echo "  type-check    Run type checking with mypy"
	@echo "  test          Run tests with pytest"
	@echo "  pre-commit    Run pre-commit hooks on all files"
	@echo "  clean         Clean up temporary files"
	@echo "  ci-local      Run all CI checks locally"

# Development environment setup
setup-dev: install-dev pre-commit
	@echo "Development environment setup complete!"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Code formatting
format:
	@echo "Formatting code..."
	ruff check . --fix
	ruff format .
	isort .

format-check:
	@echo "Checking code formatting..."
	ruff check .
	ruff format --check .
	isort --check-only --diff .

# Linting
lint:
	@echo "Running linting checks..."
	ruff check .
	@echo "Linting complete!"

# Type checking
type-check:
	@echo "Running type checks..."
	mypy .
	@echo "Type checking complete!"


# Testing
test:
	@echo "Running tests..."
	pytest --cov=. --cov-report=term-missing --cov-report=html
	@echo "Tests complete!"



# Pre-commit setup and run
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	rm -f bandit-report.json complexity-report.json mypy-report.xml coverage.xml
	@echo "Cleanup complete!"

# Run all CI checks locally
ci-local: format-check lint type-check test
	@echo "All CI checks passed locally!"

# Quick development checks (faster subset)
dev-check: format lint
	@echo "Quick development checks complete!"
