PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup ingest bronze silver analyze-reference infer-lineage generate evaluate publish test lint check demo

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

ingest:
	$(PYTHON) scripts/ingest_sources.py

bronze:
	$(PYTHON) scripts/build_bronze.py

silver:
	$(PYTHON) scripts/build_silver.py

analyze-reference:
	$(PYTHON) scripts/analyze_reference.py

infer-lineage:
	$(PYTHON) scripts/infer_lineage.py

generate:
	$(PYTHON) scripts/generate_qa.py

evaluate:
	$(PYTHON) scripts/evaluate_qa.py

publish:
	$(PYTHON) scripts/publish_snapshot.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: lint test

demo: ingest bronze
