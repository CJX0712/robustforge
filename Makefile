.PHONY: install test lint format benchmark run clean

install:
	pip install -e ".[dev]"

test:
	pytest -q -W ignore::UserWarning

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

benchmark:
	python -m robustforge.cli run --out benchmark.json

run: benchmark

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
