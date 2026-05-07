#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== Morphology Demo setup & run =="

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

if [ ! -d ".venv" ]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

echo "Activating venv..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies (this may take a while)..."
python -m pip install -r requirements.txt

echo "Starting Streamlit. Open http://localhost:8501"
streamlit run app.py --server.port 8501

