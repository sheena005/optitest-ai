#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_STREAMLIT="$PROJECT_ROOT/.venv/bin/streamlit"

echo "========================================="
echo "OptiTest AI: Launching Streamlit Dashboard"
echo "========================================="

"$VENV_STREAMLIT" run "$PROJECT_ROOT/app.py" --server.port 8501 --server.headless false
