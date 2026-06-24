#!/bin/bash
set -e 

COMPOSE_FILE="/datasets/projects/RAG/compose.yaml"

# Stop old containers
docker compose -f "$COMPOSE_FILE" down

: '
# ------- PARSE PDF ----------
echo "MinerU starting, waiting for api up"
docker compose -f "$COMPOSE_FILE" up -d --wait mineru-api

echo "MinerU ready, parsing starting"
#python3 -m src.ingest.parse_docs

echo "Shutting down mineru container"
docker compose -f "$COMPOSE_FILE" stop mineru-api
sleep 5

# --------- CHUNK ------------
echo "vLLM starting, wating for api up"
docker compose -f "$COMPOSE_FILE" up -d --wait vllm-server

echo "vLLM ready -> chunking + prefix gen starting"
#python3 -m src.ingest.gen_chunks
'

# ---------- GEN NODES/EDGES GRAPH----------
echo "Check vllm up"
docker compose -f "$COMPOSE_FILE" up -d --wait vllm-server

echo "Building graph nodes"
python3 -m src.graph.build_graph

echo "Extracting entities"
python3 -m src.graph.extract

echo "Writing nodes to graph"
python3 -m src.graph.semantic_nodes

echo "Shutting down vLLM"
docker compose -f "$COMPOSE_FILE" stop vllm-server
sleep 5

# ---------- VECTOR DB ----------
echo "Qdrant starting, waitinr for api up"
docker compose -f "$COMPOSE_FILE" up -d --wait qdrant

echo "Qdrant ready, embedding to vector store starting"
python3 -m src.embed_to_vec

# echo "Embedding summaries, entities relationships, communities..."
# python3 -m src.embed