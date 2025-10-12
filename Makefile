# VolBoost Makefile

.PHONY: setup install test run_demo clean help

# Default target
help:
	@echo "VolBoost - Options Volatility Surface ML Strategy"
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Create conda environment and install dependencies"
	@echo "  install   - Install the package in development mode"
	@echo "  test      - Run the test suite"
	@echo "  run_demo  - Run the main demonstration"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"

# Setup environment
setup:
	@echo "Setting up VolBoost environment..."
	conda env create -f env.yml
	@echo "Environment created. Activate with: conda activate vol-ml-vega"

# Install package
install:
	@echo "Installing VolBoost in development mode..."
	pip install -e .

# Run tests
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v --tb=short

# Run demo
run_demo:
	@echo "Running VolBoost demonstration..."
	python main.py

# Clean up
clean:
	@echo "Cleaning up generated files..."
	rm -rf reports/*.csv
	rm -rf reports/*.html
	rm -rf reports/*.pdf
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -delete
	find . -name ".coverage" -delete

# Development setup
dev: setup install
	@echo "Development environment ready!"

# Full test suite with coverage
test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	black --check src/ tests/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Type checking
type-check:
	@echo "Running type checks..."
	mypy src/ --ignore-missing-imports

# Security check
security:
	@echo "Running security checks..."
	bandit -r src/ -f json -o reports/security_report.json

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html

# All checks
check: lint type-check test
	@echo "All checks passed!"

# CI pipeline
ci: setup install check
	@echo "CI pipeline completed successfully!"
