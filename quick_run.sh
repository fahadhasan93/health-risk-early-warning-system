#!/usr/bin/env bash
# Quick runner for HREWS using the project's `.venv`
# Usage: ./quick_run.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Prefer .venv if present, otherwise try hrews_env
if [ -d ".venv" ]; then
  VENV_DIR=".venv"
elif [ -d "hrews_env" ]; then
  VENV_DIR="hrews_env"
else
  echo "❌ No virtual environment found."
  echo "Create one once with: python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

STREAMLIT_BIN="$VENV_DIR/bin/streamlit"

if [ ! -x "$STREAMLIT_BIN" ]; then
  # Try pip-installed entrypoint fallback
  if [ -f "$VENV_DIR/bin/python" ]; then
    echo "Installing streamlit into the venv entrypoint if missing..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
    "$VENV_DIR/bin/python" -m pip install streamlit >/dev/null
  fi
fi

echo "🚀 Launching HREWS using $VENV_DIR..."
"$STREAMLIT_BIN" run app.py --server.port 8501 --server.headless true
