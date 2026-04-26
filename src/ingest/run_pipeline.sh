# Stop if scripts fail
set -e 

# Start mineru docker container (takes a while to be ready, wait for completion?)
docker compose up -d mineru-api

#Send PDF files to mineru client to parse, write output block files 
echo "Starting parse"
python parse_docs.py

# Free up vram
echo "Shutting down mineru container"
docker compose -f /datasets/mineru/docker-compose.yml down      # this should be /datasets/projects/rag/compose.yaml down

# wait for close 
sleep 5

# Start VLLM container (LLM for contextual prefix)

# Read content_list file, generate and write chunks
echo "Starting chunking"
python gen_chunks.py

# Read chunk json, 

# embedd summaries, entities relationships, communities
python embed.py
