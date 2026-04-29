import requests
from pathlib import Path
import json
import logging
import time
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Setup config
LLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

CHUNKS_DIR = Path("/datasets/scratch/02_chunks")
OUTPUT_PATH = Path("/datasets/scratch/04_entities/entities.jsonl")

MAX_WORKERS = 6
MAX_RETRIES = 3

GRAPH_SCHEMA = {
    "type": "object",
    "properties": {

        "chunk_summary": {
            "type": "string", 
            "description": "A dense, 1-to-2 sentence summary of the entire chunk's technical content."
        },

        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {

                    "name": {
                        "type": "string",
                        "description": "The name of the concept. MUST be standard English words. MUST NOT contain LaTeX, math variables, or symbols like _, ^, or \\.",
                        "maxLength": 40 
                    },

                    "type": {
                        "type": "string",
                        "enum": ["Algorithm", "Architecture", "Metric", "Task", "Dataset", "MathematicalConcept"]
                    },
                    
                    "description": {
                        "type": "string", 
                        "description": "A concise 15-word definition of the entire entity's role in this specific text."
                    }

                },
                "required": ["name", "type", "description"]
            }
        },
        
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "description": "The relationship (e.g., USES, AFFECTS)"}
                },
                "required": ["source", "target", "type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["chunk_summary", "entities", "relationships"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are a specialized Knowledge Graph Extraction agent.
### EXTRACTION RULES:
1. CONCEPTUAL ABSTRACTION: Do not create entities for individual variables.
2. TYPE RESTRICTION: Entity types must be short, PascalCase.
3. DESCRIPTIONS: Provide a brief summary of the chunk, and a 15-word description for each entity.
4. NO MARKDOWN: Output raw JSON only."""

EMPTY_RESULT = {"chunk_summary": "", "entities": [], "relationships": []}



def extract_entities(chunk: dict) -> dict:
    """Call the LLM to extract entities and relationships from a single chunk.
    
    Retries up to MAX_RETRIES times on transient failures.
    Returns EMPTY_RESULT shape on permanent failure.
    """
    chunk_text = chunk.get("text", "").strip()
    if not chunk_text:
        return dict(EMPTY_RESULT)
    
    prompt = (
        f"Analyze the following text and extract the key entities and relationships.\n\n"
        f"Text: {chunk_text}"
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.5,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "graph_extraction",
                "schema": GRAPH_SCHEMA,
                "strict": True,
            },
        },
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                LLM_URL,
                headers={"Authorization": "Bearer EMPTY"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result_text = response.json()["choices"][0]["message"]["content"]
            return json.loads(result_text)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed extraction for chunk {chunk.get('chunk_id')}: {e}")
                return dict(EMPTY_RESULT)
            time.sleep(2 ** attempt)

def load_done_chunk_ids(output_path: Path) -> set:
    """Read existing entities.jsonl to find chunk_ids already processed."""
    done = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def collect_pending_chunks(chunks_dir: Path, done: set) -> list:
    """Walk all chunk files and return chunks not yet processed."""
    pending = []
    for json_file in sorted(chunks_dir.glob("*.json")):
        chunks = json.loads(json_file.read_text(encoding="utf-8"))
        for chunk in chunks:
            if chunk["chunk_id"] not in done:
                pending.append(chunk)
    return pending


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    done = load_done_chunk_ids(OUTPUT_PATH)
    if done:
        logger.info(f"Found {len(done)} already-extracted chunks, will skip")
    
    tasks = collect_pending_chunks(CHUNKS_DIR, done)
    logger.info(f"Processing {len(tasks)} chunks")
    
    if not tasks:
        logger.info("Nothing to do.")
        return
    
    # Append-mode + flush after each write makes every success durable
    with OUTPUT_PATH.open("a") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_entities, c): c for c in tasks}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                chunk = futures[future]
                try:
                    extraction = future.result()
                    record = {"chunk_id": chunk["chunk_id"], "extraction": extraction}
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    
                    if i % 100 == 0:
                        logger.info(f"[{i}/{len(tasks)}] processed")
                
                except Exception as e:
                    logger.error(f"Worker exception for {chunk['chunk_id']}: {e}")
    
    logger.info("Entity extraction complete")


if __name__ == "__main__":
    main()