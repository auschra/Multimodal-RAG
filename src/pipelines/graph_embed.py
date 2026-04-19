from src.clients.embedding_client import EmbeddingClient
import networkx as nx
from src.config import load_config
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from pathlib import Path
import time

cfg = load_config()

# Load graph
graph_path = Path(cfg.dirs.processed_data) / "knowledge_graph.graphml"
graph = nx.read_graphml(str(graph_path))
embedder = EmbeddingClient(device='cuda')

time0 = time.time()

for node, data in graph.nodes(data=True):
    t = str(data.get("type", "")).strip().lower()

    if t == "entity":
        ent_label = data.get("label")
        ent_descr = data.get("description")         # Fix 2 chunks with misisng vals
        if ent_descr:
            emb_ent_descr = embedder.embed_text(ent_descr)

    elif t == "chunk":
        chunk_id = data.get("vec_id")
        chunk_sum = data.get("summary")
        if not chunk_sum:
            print(f"node {node}: missing 'summary', keys={list(data.keys())}")
            continue
        emb_chunk_sum = embedder.embed_text(chunk_sum)

    else:  # document or unknown
        pass

time1 = time.time()
print(f"{time1 - time0:.2f}")

""" 
2 chunks missing data 
Skipping node chunk_81ca00b5: missing 'summary', keys=['type', 'chunk_type', 'text', 'heading', 'vec_id', 'page', 'bbox']
Skipping node chunk_7e73866b: missing 'summary', keys=['type', 'chunk_type', 'text', 'heading', 'vec_id', 'page', 'bbox']

type — unique: 3, total occurrences: 1824
 - entity: 1514
 - Chunk: 308
 - Document: 2

category — unique: 6, total occurrences: 1514
 - MathematicalConcept: 668
 - Architecture: 380
 - Algorithm: 222
 - Dataset: 135
 - Task: 83
 - Metric: 26

chunk_type — unique: 7, total occurrences: 308
 - text: 223
 - list_item: 56
 - caption: 10
 - document_index: 5
 - table: 5
 - formula: 5
 - code: 4
 
 
 Type: chunk — variants: Chunk (308) — unique keys: 8
 - bbox
 - chunk_type
 - heading
 - page
 - summary
 - text
 - type
 - vec_id

Type: document — variants: Document (2) — unique keys: 3
 - filepath
 - title
 - type

Type: entity — variants: entity (1514) — unique keys: 4
 - category
 - description
 - label
 - type
 """