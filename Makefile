# NetLab Development Makefile

.PHONY: help venv clean-venv dev install check check-ci lint format test qt build clean check-dist publish-test publish info hooks check-python

.DEFAULT_GOAL := help

# --------------------------------------------------------------------------
# Python interpreter detection
# --------------------------------------------------------------------------
# VENV_BIN: path to local virtualenv bin directory
VENV_BIN := $(PWD)/venv/bin

# PY_BEST: scan for newest supported Python (used when creating new venvs)
# Supports 3.11-3.13 to match common CI matrix
PY_BEST := $(shell for v in 3.13 3.12 3.11; do command -v python$$v >/dev/null 2>&1 && { echo python$$v; exit 0; }; done; command -v python3 2>/dev/null || command -v python 2>/dev/null)

# PY_PATH: active python3/python on PATH (respects CI setup-python and activated venvs)
PY_PATH := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

# PYTHON: interpreter used for all commands
#   1. Use local venv if present
#   2. Otherwise use active python on PATH (important for CI)
#   3. Fall back to best available version
#   4. Final fallback to 'python3' literal for clear error messages
PYTHON ?= $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,$(if $(PY_PATH),$(PY_PATH),$(if $(PY_BEST),$(PY_BEST),python3)))

# Derived tool commands (always use -m to ensure correct environment)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
PRECOMMIT := $(PYTHON) -m pre_commit

help:
	@echo "🔧 NetLab Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make venv          - Create a local virtualenv (./venv)"
	@echo "  make dev           - Full development environment (package + dev deps + hooks)"
	@echo "  make install       - Install package for usage (no dev dependencies)"
	@echo "  make clean-venv    - Remove virtual environment"
	@echo ""
	@echo "Code Quality & Testing:"
	@echo "  make check         - Run pre-commit (auto-fix) + tests, then lint"
	@echo "  make check-ci      - Run non-mutating lint + tests (CI entrypoint)"
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
	@echo "  make hooks         - Run pre-commit on all files"
	@echo "  make check-python  - Check if venv Python matches best available"

dev:
	@echo "🚀 Setting up development environment..."
	@if [ ! -x "$(VENV_BIN)/python" ]; then \
		if [ -z "$(PY_BEST)" ]; then \
			echo "❌ Error: No Python interpreter found (python3 or python)"; \
			exit 1; \
		fi; \
		echo "🐍 Creating virtual environment with $(PY_BEST) ..."; \
		$(PY_BEST) -m venv venv || { echo "❌ Failed to create venv"; exit 1; }; \
		if [ ! -x "$(VENV_BIN)/python" ]; then \
			echo "❌ Error: venv creation failed - $(VENV_BIN)/python not found"; \
			exit 1; \
		fi; \
		$(VENV_BIN)/python -m pip install -U pip setuptools wheel; \
	fi
	@echo "📦 Installing dev dependencies..."
	@$(VENV_BIN)/python -m pip install -e .'[dev]'
	@echo "🔗 Installing pre-commit hooks..."
	@$(VENV_BIN)/python -m pre_commit install --install-hooks
	@echo "✅ Dev environment ready. Activate with: source venv/bin/activate"
	@$(MAKE) check-python

venv:
	@echo "🐍 Creating virtual environment in ./venv ..."
	@if [ -z "$(PY_BEST)" ]; then \
		echo "❌ Error: No Python interpreter found (python3 or python)"; \
		exit 1; \
	fi
	@$(PY_BEST) -m venv venv || { echo "❌ Failed to create venv"; exit 1; }
	@if [ ! -x "$(VENV_BIN)/python" ]; then \
		echo "❌ Error: venv creation failed - $(VENV_BIN)/python not found"; \
		exit 1; \
	fi
	@$(VENV_BIN)/python -m pip install -U pip setuptools wheel
	@echo "✅ venv ready. Activate with: source venv/bin/activate"

clean-venv:
	@rm -rf venv/

install:
	@echo "📦 Installing package for usage (no dev dependencies)..."
	@$(PIP) install -e .

check:
	@echo "🔍 Running complete code quality checks and tests..."
	@PYTHON=$(PYTHON) bash dev/run-checks.sh
	@$(MAKE) lint

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
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml coverage-*.xml .benchmarks .pytest-benchmark || true
	@find . -path "./venv" -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*~" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*.orig" -delete 2>/dev/null || true
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
	@echo "  Python (active): $$($(PYTHON) --version)"
	@echo "  Python (best):   $$($(PY_BEST) --version 2>/dev/null || echo 'missing')"
	@$(MAKE) check-python
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

hooks:
	@echo "🔗 Running pre-commit on all files..."
	@$(PRECOMMIT) run --all-files || (echo "Some pre-commit hooks failed. Fix and re-run." && exit 1)

check-python:
	@if [ -x "$(VENV_BIN)/python" ]; then \
		VENV_VER=$$($(VENV_BIN)/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown"); \
		BEST_VER=$$($(PY_BEST) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown"); \
		if [ -n "$$VENV_VER" ] && [ -n "$$BEST_VER" ] && [ "$$VENV_VER" != "$$BEST_VER" ]; then \
			echo "⚠️  WARNING: venv Python ($$VENV_VER) != best available Python ($$BEST_VER)"; \
			echo "   Run 'make clean-venv && make dev' to recreate venv if desired"; \
		fi; \
	fi
