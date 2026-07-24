PYTHON := .venv/bin/python
PIP := .venv/bin/pip
AS_OF_DATE ?= 2026-07-23

.PHONY: setup ingest test lint check

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

ingest:
	$(PYTHON) scripts/ingest_sources.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test

bronze:
	$(PYTHON) scripts/extract_bronze.py