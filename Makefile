PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip

.PHONY: install lint test run-pipeline format

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests
	$(VENV)/bin/isort --check-only src tests

test:
	PYTHONPATH=src $(VENV)/bin/pytest -q

run-pipeline:
	PYTHONPATH=src $(VENV)/bin/python -m pipeline

format:
	$(VENV)/bin/black src tests
	$(VENV)/bin/isort src tests
