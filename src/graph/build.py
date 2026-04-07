import networkx as nx
import uuid
import json
import requests


class BuildGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # Create parent doc node
    def intake_text(self, title, metadata):
        doc_node_id = f"doc_{uuid.uuid4().hex[:8]}"
        self.G.add_node(doc_node_id, type="Document", title=title, **metadata)                          # Create doc node

        return doc_node_id
    
   
    def assign_node(self, doc_node_id, chunk_text, vec_id, page_no, chunk_type, bbox, heading):         # Create child chunk node and assign to parent doc
        chunk_node_id = f"chunk_{uuid.uuid4().hex[:8]}"
        safe_bbox = json.dumps(bbox) if bbox else ""
        safe_vec = str(vec_id) if vec_id else ""

        self.G.add_node(
            chunk_node_id, 
            type="Chunk", 
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
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The name of the entity"},
                    "type": {
                        "type": "string", 
                        "enum": ["MathematicalObject", "Algorithm", "FunctionSpace", "Theorem", "ComplexityClass", "Dataset", "Operator", "Hardware", "ResearchField", "Metric"],
                        "description": "High-level category"
                    },
                    "description": {"type": "string", "description": "15-word max role of the entity"}
                },
                "required": ["id", "type", "description"],
                "additionalProperties": False
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
    "required": ["entities", "relationships"],
    "additionalProperties": False
}


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
                    "content": ("""You are a specialized Knowledge Graph Extraction agent for high-level Scientific and Mathematical papers.
                                    Your goal is to extract a clean, queryable topology of concepts, skipping raw algebraic derivations.

                                    ### EXTRACTION RULES:
                                    1. CONCEPTUAL ABSTRACTION: Do not create entities for individual variables (e.g., "alpha", "beta", "x_prime"). 
                                    - Instead, group them under the parent concept (e.g., "Linear Signal Action").
                                    2. TYPE RESTRICTION: Entity types must be short, PascalCase, and high-level (e.g., "Manifold", "Operator", "FunctionSpace", "Dataset").
                                    - NEVER use LaTeX or long descriptions in the "type" field.
                                    3. DESCRIPTION BREVITY: Limit "description" to 15 words. Focus on the entity's role in the paper.
                                    4. NO MARKDOWN: Output raw JSON only. Do not wrap in ```json blocks.

                                    ### SCHEMA STRUCTURE:
                                    - id: A short unique identifier (e.g., "Manifold_Omega").
                                    - type: A single high-level category.
                                    - description: A concise definition of the entity's function. 
                                    """
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.0,             
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
    
    
def add_semantics_to_graph(builder, chunk_node_id, json_output):

    for entity in json_output.get("entities", []):                                                      # Create new entity node from json output
        node_id = f"ent_{entity['id'].lower().replace(' ', '_')}"
        
        if not builder.G.has_node(node_id):
            builder.G.add_node(node_id, type=entity['type'], name=entity['id'])
            
        builder.G.add_edge(node_id, chunk_node_id, relationship="MENTIONED_IN")                         # Connect new entity to its chunk

    for rel in json_output.get("relationships", []):                                        
        source_id = f"ent_{rel['source'].lower().replace(' ', '_')}"
        target_id = f"ent_{rel['target'].lower().replace(' ', '_')}"
        
        if builder.G.has_node(source_id) and builder.G.has_node(target_id):                             # Create relationship edge if necessary
            builder.G.add_edge(source_id, target_id, relationship=rel['type'])

            