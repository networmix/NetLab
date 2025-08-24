#!/bin/bash
# Run all code quality checks and tests
# This script runs pre-commit hooks and the test suite

set -e  # Exit on any error

# Determine python interpreter (prefer venv if active)
PYTHON=${PYTHON:-python3}

# Check if pre-commit is installed
if ! "$PYTHON" -m pre_commit --version &> /dev/null; then
    echo "❌ pre-commit is not installed. Please run 'make dev' first."
    exit 1
fi

# Check if pytest is installed
if ! "$PYTHON" -m pytest --version &> /dev/null; then
    echo "❌ pytest is not installed. Please run 'make dev' first."
    exit 1
fi

# Check if pre-commit hooks are installed
if [ ! -f .git/hooks/pre-commit ]; then
    echo "⚠️  Pre-commit hooks not installed. Installing now..."
    "$PYTHON" -m pre_commit install
    echo ""
fi

# Run pre-commit with fixers (first pass), do not fail if files were modified
echo "🏃 Running pre-commit (first pass: apply auto-fixes if needed)..."
set +e
"$PYTHON" -m pre_commit run --all-files
first_pass_status=$?
set -e

if [ $first_pass_status -ne 0 ]; then
    echo "ℹ️  Some hooks modified files or reported issues. Re-running checks..."
fi

# Re-run to verify all checks pass after fixes; fail on any remaining issues
echo "🏃 Running pre-commit (second pass: verify all checks)..."
if ! "$PYTHON" -m pre_commit run --all-files; then
    echo ""
    echo "❌ Pre-commit checks failed after applying fixes. Please address the issues above."
    exit 1
fi

# Track whether auto-fixes were applied and resolved issues
autofixed=0
if [ $first_pass_status -ne 0 ]; then
    autofixed=1
fi

echo ""
echo "✅ Pre-commit checks passed!"
echo ""

echo ""

# Run tests with coverage (includes slow and benchmark tests for regression detection)
echo "🧪 Running tests with coverage..."
"$PYTHON" -m pytest

if [ $? -eq 0 ]; then
    echo ""
    if [ $autofixed -eq 1 ]; then
        echo "🎉 All checks and tests passed. Auto-fixes were applied by pre-commit."
    else
        echo "🎉 All checks and tests passed."
    fi
else
    echo ""
    if [ $autofixed -eq 1 ]; then
        echo "❌ Some tests failed. Note: auto-fixes were applied earlier by pre-commit."
    else
        echo "❌ Some tests failed."
    fi
    exit 1
fi
