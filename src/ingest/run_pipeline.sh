#!/bin/bash
set -e 

COMPOSE_FILE="/datasets/projects/RAG/compose.yaml"
VLLM_BASE_URL="http://localhost:8001/v1"
VLLM_API_KEY="EMPTY"

# Stop old containers
docker compose -f $COMPOSE_FILE down

: '
# ------- PARSE PDF ----------
# Start mineru docker container (takes a while to be ready, wait for completion?)
docker compose -f $COMPOSE_FILE down mineru-api 2>/dev/null || true
docker compose -f $COMPOSE_FILE up -d mineru-api


# Wait for mineru
echo "Waiting for mineru client"


for i in {1..60}; do
    if curl -sf http://localhost:8000/docs > /dev/null 2>&1; then
        echo "MinerU is ready"
        break
    fi
    sleep 5
    if [ $i -eq 60 ]; then
        echo "ERROR: MinerU did not become ready within 5 minutes" >&2
        docker compose -f $COMPOSE_FILE logs mineru-api
        exit 1
    fi
done
'

sleep 5

#Send PDF files to mineru client to parse, write output block files 
echo "Starting parse"
#python3 -m src.ingest.parse_docs

# Free up vram
echo "Shutting down mineru container"
#docker compose -f $COMPOSE_FILE stop

# wait for close 
sleep 5

# --------- CHUNK ------------
# Start VLLM container (LLM for contextual prefix)
echo "Booting vllm server"
docker compose -f "$COMPOSE_FILE" up -d vllm-server

# Stream startup logs while the readiness probe runs.
docker compose -f "$COMPOSE_FILE" logs -f --tail=50 vllm-server &
VLLM_LOG_PID=$!

echo "Waiting for vLLM to be ready"
for i in {1..60}; do
    if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_BASE_URL/models" > /dev/null 2>&1; then
        echo "vLLM is ready"
        kill "$VLLM_LOG_PID" 2>/dev/null || true
        break
    fi
    sleep 5
    if [ "$i" -eq 60 ]; then
        echo "ERROR: vLLM did not become ready within 5 minutes" >&2
        kill "$VLLM_LOG_PID" 2>/dev/null || true
        docker compose -f "$COMPOSE_FILE" logs vllm-server
        exit 1
    fi
done

: '
# Read content_list file, generate and write chunks
echo "Starting chunking and prefix gen"
python3 -m src.ingest.gen_chunks
'
# ---------- GEN NODES/EDGES ----------


echo "Stopping vLLM to free VRAM..."
docker compose -f $COMPOSE_FILE stop vllm-server
sleep 5



# embedd summaries, entities relationships, communities
#python embed
