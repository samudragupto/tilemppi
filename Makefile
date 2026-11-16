.PHONY: install lint typecheck test verify-cutile benchmark plots frontend serve ci clean

install:
	pip install -e ".[server,dev]"

lint:
	ruff format --check .
	ruff check .
	bandit -c pyproject.toml -r src/tilemppi

typecheck:
	mypy src/tilemppi

test:
	pytest --cov=tilemppi --cov-report=term-missing -v

verify-cutile:
	tilemppi verify-cutile

benchmark:
	tilemppi benchmark --quick

plots:
	tilemppi export-plots

frontend:
	cd frontend && npm ci && npm run lint && npm run typecheck && npm run test && npm run build

serve:
	tilemppi serve --host 127.0.0.1 --port 8000

ci:
	python scripts/ci_check.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
