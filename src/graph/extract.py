import requests
from pathlib import Path
import json
import concurrent.futures

llm_url = "http://localhost:8001/v1/chat/completions"
llm_model = "Qwen/Qwen3-32B-AWQ"
chunk_dir = Path("/datasets/scratch/02_chunks")
out_path = Path("/datasets/scratch/04_entities/entities.jsonl")

# Force generator to output json summaries, entities and relationships.
graph_schema = {
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

system_prompt = """You are a specialized Knowledge Graph Extraction agent.
Extract key entities (nodes) and their relationships (edges) from the provided text.

### EXTRACTION RULES:
1. CONCEPTUAL ABSTRACTION: Do not create entities for individual variables.
2. GRANULARITY: Entities must be atomic (e.g., "Python", not "The Python programming language").
3. ENTITY RESOLUTION: Normalize entity names. Resolve pronouns and acronyms to their full forms.
4. TYPE RESTRICTION: Assign a broad category to each entity. Types must be short, PascalCase (e.g., Person, Organization, Technology).
5. EDGE LABELS: Relationships must be directed and labeled with concise, UPPERCASE_SNAKE_CASE descriptors (e.g., FOUNDED_BY, DEPENDS_ON).
6. DESCRIPTIONS: Provide exactly a 15-word description for each entity.
7. STRICT GROUNDING: Only extract entities and relationships explicitly stated or directly implied in the current chunk. Do not hallucinate external knowledge."""

def extract_entities(chunk):
    chunk_text = chunk.get("text", "").strip()
    prompt = f"Analyze the following text:\n\n{chunk_text}"

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.0, 
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "graph_extraction",
                "schema": graph_schema,
                "strict": True,
            },
        },
    }

    # Query llm, return text output
    response = requests.post(
        llm_url,
        headers={"Authorization": "Bearer EMPTY"},
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    result_text = response.json()["choices"][0]["message"]["content"]
    return json.loads(result_text)
        

# Check which chunks have been extracted
def check_done_chunks(output_path):
    done = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["chunk_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done

# Check which chunks havent been extracted yet
def check_new_chunks(chunks_dir, done):
    new = []
    for json_file in sorted(chunks_dir.glob("*.json")):
        chunks = json.loads(json_file.read_text(encoding="utf-8"))
        
        for chunk in chunks:
            if chunk["chunk_id"] not in done:
                new.append(chunk)
    return new


def main():
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check done and new chunks
    done = check_done_chunks(out_path)
    tasks = check_new_chunks(chunk_dir, done)
    
    # Start the extraction process for new chunks
    with out_path.open("a") as out:

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(extract_entities, c): c for c in tasks}      # create future jobs for new chunks
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                chunk = futures[future]

                extraction = future.result()
                record = {"chunk_id": chunk["chunk_id"], "extraction": extraction}
                out.write(json.dumps(record) + "\n")
                out.flush()

if __name__ == "__main__":
    main()