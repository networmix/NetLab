# NetLab Development Makefile

.PHONY: help dev install check check-ci lint format test qt clean build check-dist publish-test publish info

.DEFAULT_GOAL := help

VENV_BIN := $(PWD)/netlab-venv/bin
PYTHON := $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,python3)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
PRECOMMIT := $(PYTHON) -m pre_commit

help:
	@echo "🔧 NetLab Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install package for usage (no dev dependencies)"
	@echo "  make dev           - Full development environment (package + dev deps + hooks)"
	@echo ""
	@echo "Code Quality & Testing:"
	@echo "  make check         - Run lint + tests"
	@echo "  make check-ci      - Run non-mutating checks and tests (CI entrypoint)"
	@echo "  make lint          - Run only linting (non-mutating: ruff + pyright)"
	@echo "  make format        - Auto-format code with ruff"
	@echo "  make test          - Run tests with coverage"
	@echo "  make qt            - Run quick tests only (exclude benchmark)"
	@echo ""
	@echo "Build & Package:"
	@echo "  make build         - Build distribution packages"
	@echo "  make clean         - Clean build artifacts and cache files"
	@echo "  make check-dist    - Check distribution packages with twine"
	@echo "  make publish-test  - Publish to Test PyPI"
	@echo "  make publish       - Publish to PyPI"
	@echo ""
	@echo "Utilities:"
	@echo "  make info          - Show project information"

dev:
	@echo "🚀 Setting up development environment..."
	@bash dev/setup-dev.sh

install:
	@echo "📦 Installing package for usage (no dev dependencies)..."
	@$(PIP) install -e .

check:
	@echo "🔍 Running complete code quality checks and tests..."
	@$(MAKE) lint
	@PYTHON=$(PYTHON) bash dev/run-checks.sh

check-ci:
	@echo "🔍 Running CI checks (non-mutating lint + tests)..."
	@$(MAKE) lint
	@$(MAKE) test

lint:
	@echo "🧹 Running linting checks (non-mutating)..."
	@$(RUFF) format --check .
	@$(RUFF) check .
	@$(PYTHON) -m pyright

format:
	@echo "✨ Auto-formatting code..."
	@$(RUFF) format .

test:
	@echo "🧪 Running tests with coverage..."
	@$(PYTEST)

qt:
	@echo "⚡ Running quick tests only (exclude benchmark)..."
	@$(PYTEST) --no-cov -m "not benchmark"

build:
	@echo "🏗️  Building distribution packages..."
	@if $(PYTHON) -c "import build" >/dev/null 2>&1; then \
		$(PYTHON) -m build; \
	else \
		echo "❌ build module not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

clean:
	@echo "🧹 Cleaning build artifacts and cache files..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*~" -delete
	@find . -type f -name "*.orig" -delete
	@echo "✅ Cleanup complete!"

check-dist:
	@echo "🔍 Checking distribution packages..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine check dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

publish-test:
	@echo "📦 Publishing to Test PyPI..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine upload --repository testpypi dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

publish:
	@echo "🚀 Publishing to PyPI..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine upload dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

info:
	@echo "📋 NetLab Project Information"
	@echo "============================="
	@echo ""
	@echo "🐍 Python Environment:"
	@echo "  Python version: $$($(PYTHON) --version)"
	@echo "  Package version: $$($(PYTHON) -c 'import importlib.metadata as m; print(m.version("netlab"))' 2>/dev/null || echo 'Not installed')"
	@echo "  Virtual environment: $$(echo $$VIRTUAL_ENV | sed 's|.*/||' || echo 'None active')"
	@echo ""
	@echo "🔧 Development Tools:"
	@echo "  Pre-commit: $$($(PRECOMMIT) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pytest: $$($(PYTEST) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Ruff: $$($(RUFF) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pyright: $$($(PYTHON) -m pyright --version 2>/dev/null | head -1 || echo 'Not installed')"
	@echo "  Build: $$($(PYTHON) -m build --version 2>/dev/null | sed 's/build //' | sed 's/ (.*//' || echo 'Not installed')"
	@echo "  Twine: $$($(PYTHON) -m twine --version 2>/dev/null | grep -o 'twine version [0-9.]*' | cut -d' ' -f3 || echo 'Not installed')"
