#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$PROJECT_ROOT/.venv/bin/python3"
VENV_PYTEST="$PROJECT_ROOT/.venv/bin/pytest"

echo "========================================="
echo "OptiTest AI: Full Verification Suite"
echo "========================================="

echo ""
echo "[1/2] Executing Native C++ Binary Tests..."
"$PROJECT_ROOT/build/optitest_cpp_tests"

echo ""
echo "[2/2] Executing Python Pytest Automation Suite..."
"$VENV_PYTEST" "$PROJECT_ROOT/tests" -v --cov=optitest

echo ""
echo "========================================="
echo "ALL TESTS PASSED SUCCESSFULLY!"
echo "========================================="
