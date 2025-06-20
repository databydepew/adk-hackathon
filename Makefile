# Makefile for Vertex AI Pipeline Agent

.PHONY: help clean clean-pyc clean-build clean-test venv install install-dev run run-interactive run-single run-batch build package

# Default target
help:
	@echo "Vertex AI Pipeline Agent Makefile"
	@echo "--------------------------------"
	@echo "setup       - create virtual environment and install dependencies"
	@echo "install     - install dependencies"
	@echo "install-dev - install development dependencies"
	@echo "run         - run in interactive mode"
	@echo "run-single  - run with a single instruction"
	@echo "run-batch   - run with batch instructions from a file"
	@echo "clean       - remove all build, test, and Python artifacts"
	@echo "clean-pyc   - remove Python file artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-test  - remove test and coverage artifacts"
	@echo "build       - build the package"
	@echo "package     - create source and wheel distributions"

# Environment setup
venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

setup: venv
	. venv/bin/activate && pip install -e .
	@echo "Environment setup complete. Activate with: source venv/bin/activate"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Run targets
run:
	python -m src.main --interactive

run-single:
	@if [ -z "$(INSTRUCTION)" ]; then \
		echo "Error: INSTRUCTION not provided. Use 'make run-single INSTRUCTION=\"your instruction\"'"; \
		exit 1; \
	fi
	python -m src.main --instruction "$(INSTRUCTION)"

run-batch:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not provided. Use 'make run-batch FILE=your_file.txt'"; \
		exit 1; \
	fi
	python -m src.main --batch $(FILE)

# Build and package
build:
	python3 setup.py build

package:
	python3 setup.py sdist bdist_wheel

# Clean targets
clean: clean-pyc clean-build clean-test
	rm -rf venv/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.so' -delete
	find . -name '.coverage' -delete
	find . -name '.coverage.*' -delete

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/

clean-test:
	rm -rf .tox/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .hypothesis/
