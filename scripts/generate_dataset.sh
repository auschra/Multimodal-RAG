#!/bin/bash
set -e 

COMPOSE_FILE="docker-compose.yml"

echo "Starting llm container,"
# The --wait flag replaces your entire 60-second polling loop
docker compose -f "$COMPOSE_FILE" up -d --wait vllm-server

echo "vllm ready, starting dataset generation"
# Run the Python evaluation script
python -m evals.gen_dataset --mode=single

echo "dataset creation complete."