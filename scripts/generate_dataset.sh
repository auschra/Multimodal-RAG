#!/bin/bash
set -e 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/compose.yaml"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

echo "Starting llm container,"
# The --wait flag replaces your entire 60-second polling loop
docker compose -f "$COMPOSE_FILE" up -d --wait vllm-server

echo "vllm ready, starting dataset generation"
# Run the Python evaluation script
python -m evals.gen_dataset --mode=single

echo "dataset creation complete."

"$PYTHON_BIN" -m evals.gen_dataset --mode=single