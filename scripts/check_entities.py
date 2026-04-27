import networkx as nx
import random
from pathlib import Path
from src.config import load_config
import json

cfg = load_config()
processed = cfg.dirs.processed_data

graph_path = f"{processed}/knowledge_graph.graphml"
G = nx.read_graphml(graph_path)
entities_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'entity']

collected = []

for target_entity in entities_nodes:
    entity_data = G.nodes[target_entity]
    label = entity_data.get('label', '')

    # Go back to parent chunks mentioning entity
    for pred_id in G.predecessors(target_entity):
        pred_data = G.nodes[pred_id]

        # Skip non-chunk parents (ie documents)
        if pred_data.get('type') != 'Chunk':
            continue

        chunk_id = pred_id
        chunk_data = pred_data

        # Find the parent doc of the parent chunks 
        for doc_id in G.predecessors(chunk_id):
            doc_data = G.nodes[doc_id]
            if doc_data.get('type') != 'Document':
                continue

            record = {
                "entity_id": target_entity,
                "entity_label": label,
                "document_id": doc_id,
                "document_title": doc_data.get('title', 'Unknown Title'),
                "chunk_id": chunk_id,
                "page": chunk_data.get('page', chunk_data.get('page_no', 'N/A')),
                "heading": chunk_data.get('heading', 'Root'),
                "summary": chunk_data.get('summary'),
                "text_snippet": (chunk_data.get('text', '') or '')[:300]
            }
            collected.append(record)

# save to json
processed_path = Path(processed)
processed_path.mkdir(parents=True, exist_ok=True)
out_file = processed_path / "entities.json"

with out_file.open("w", encoding="utf-8") as f:
    json.dump(collected, f, ensure_ascii=False, indent=2)

print(f"Saved {len(collected)} entity-source records to {out_file}")