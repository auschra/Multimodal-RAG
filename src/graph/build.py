import networkx as nx
import uuid
import json
import requests


class BuildGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def intake_text(self, title, metadata):
        doc_node_id = f"doc_{title.replace(' ', '_')}"
        self.G.add_node(doc_node_id, type="Document", title=title, **metadata)                          # Create doc node

        return doc_node_id
    
    def assign_node(self, doc_node_id, chunk_text, vec_id, page_no, chunk_type, bbox, heading):      
        chunk_node_id = f"chunk_{uuid.uuid4().hex[:8]}"
        safe_bbox = json.dumps(bbox) if bbox else ""
        safe_vec = str(vec_id) if vec_id else ""

        self.G.add_node(
            chunk_node_id, 
            type="chunk", 
            chunk_type=chunk_type,
            text=chunk_text, 
            heading=heading,
            vec_id=safe_vec, 
            page=page_no,
            bbox=safe_bbox
        )
        self.G.add_edge(doc_node_id, chunk_node_id, relationship="CONTAINS")
        return chunk_node_id

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

system_prompt = """You are a specialized Knowledge Graph Extraction agent.
### EXTRACTION RULES:
1. CONCEPTUAL ABSTRACTION: Do not create entities for individual variables. 
2. TYPE RESTRICTION: Entity types must be short, PascalCase.
3. DESCRIPTIONS: Provide a brief summary of the chunk, and a 15-word description for each entity.
4. NO MARKDOWN: Output raw JSON only."""

def extract_entities(chunk_text: str):
    prompt = f"Analyze the following text and extract the key entities and relationships.\n\nText: {chunk_text}"
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},

        json={
            "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
            "messages": [
                {
                    "role": "system", 
                    "content": (system_prompt)
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.1,             
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "graph_extraction",
                    "schema": GRAPH_SCHEMA,
                    "strict": True 
                }
            }
        }
    )
    
    try:
        result_text = response.json()['choices'][0]['message']['content']   
        return json.loads(result_text)
    
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\nRaw Output: {response.text}")
        return {"entities": [], "relationships": []}
    
    
def add_semantics_to_graph(builder, chunk_node_id, extraction_data):
    
    if "chunk_summary" in extraction_data:
        builder.G.nodes[chunk_node_id]["summary"] = extraction_data["chunk_summary"]                        # Assign chunk summary to chunk node

    entity_map = {} 
    for entity in extraction_data.get("entities", []):                                                      # Create normalised entity mapping 
        
        raw_name = entity.get("name")
        clean_id = f"ent:{str(raw_name).lower().strip().replace(' ', '_')}"
        entity_map[raw_name.lower()] = clean_id
       
        builder.G.add_node(
            clean_id, 
            type="entity", 
            label=raw_name, 
            category=entity.get("type"),
            description=entity.get("description")
        )
        builder.G.add_edge(chunk_node_id, clean_id, relation="MENTIONS")

    
    for rel in extraction_data.get("relationships", []):                                                    # Create relationship
        src = rel.get("source").lower()
        tgt = rel.get("target").lower()
        
        if src in entity_map and tgt in entity_map:
            builder.G.add_edge(
                entity_map[src], 
                entity_map[tgt], 
                relation=rel.get("type").upper()
            )