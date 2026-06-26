#!/usr/bin/env bash
# ============================================================================
# SmartRoadVision — one-shot local setup script
#   1. creates a Python 3.12 virtual environment
#   2. installs all dependencies (app + dashboard + dev)
#   3. downloads model weights (YOLOv11 + SAM2 + MiDaS)
#   4. starts PostgreSQL + Redis and initialises the database schema
#   5. prints next-step run commands
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR=".venv"

echo "==> [1/5] Creating virtual environment (${VENV_DIR})"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "    ${PYTHON_BIN} not found; falling back to python3"
  PYTHON_BIN="python3"
fi
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "==> [2/5] Installing dependencies (app + dashboard + dev)"
pip install -e ".[dashboard,dev]"

echo "==> [3/5] Preparing environment file"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "    Created .env from .env.example — review and edit as needed."
fi

echo "==> [4/5] Downloading model weights"
python scripts/download_models.py --all || \
  echo "    (model download skipped/failed — system still runs with base YOLOv11)"

echo "==> [5/5] Starting PostgreSQL + Redis and initialising the database"
if command -v docker >/dev/null 2>&1; then
  docker compose up -d postgres redis
  echo "    Waiting for PostgreSQL to become healthy..."
  for _ in $(seq 1 30); do
    if docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-smartroad}" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  python - <<'PY'
import asyncio
from src.database.models import init_db
try:
    asyncio.run(init_db())
    print("    Database schema initialised.")
except Exception as exc:
    print(f"    DB init skipped: {exc}")
PY
else
  echo "    Docker not found — skipping DB/Redis startup."
fi

cat <<'EOF'

============================================================================
 SmartRoadVision setup complete.

 Activate the environment:        source .venv/bin/activate

 Run the API (dev):               uvicorn src.api.main:app --reload
 Run the dashboard:               streamlit run dashboard/app.py
 Run the full stack (Docker):     docker compose up --build
 Run the test suite:              pytest
 CLI demo:                        python scripts/run_demo.py --source data/samples/road.jpg

 API docs:   http://localhost:8000/docs
 Dashboard:  http://localhost:8501
============================================================================
EOF
