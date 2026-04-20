from src.clients.embedding_client import EmbeddingClient
import networkx as nx
from src.config import load_config
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from pathlib import Path
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

cfg = load_config()

# Load graph
graph_path = Path(cfg.dirs.processed_data) / "graph/knowledge_graph.graphml"
graph = nx.read_graphml(str(graph_path))
embedder = EmbeddingClient(device='cuda')

# Some entities are missing _ spaces ie 'graphrepresentation'

def embed_nodes():
    chunk_records = []
    entity_records = []
    
    for node, data in graph.nodes(data=True):
        node_type = str(data.get('type', '')).lower()           # chunk may exist as 'Chunk'
        
        if node_type == 'chunk':
            summary = data['text']
            embedding = embedder.embed_text(summary)
            
            chunk_records.append({
                "id": node, 
                "embedding": embedding,
                "text": summary,
                "page": data.get('page'),
                "bbox": data.get('bbox'),
                "doc_id": data.get('parent_doc')
            })

        # Entities
        elif node_type == 'entity':
            description = data['description']
            embedding = embedder.embed_text(description)
            
            entity_records.append({
                "id": node,
                "embedding": embedding,
                "name": data.get('label', node),
                "description": description,
                "category": data.get('category', 'general')
            })

    # Save
    output_dir = Path(cfg.dirs.embeddings) / "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if chunk_records:
        df_chunks = pd.DataFrame(chunk_records)
        df_chunks.to_parquet(output_dir / "chunks_embedded.parquet")
        
    if entity_records:
        df_entities = pd.DataFrame(entity_records)
        df_entities.to_parquet(output_dir / "entities_embedded.parquet")

            
embed_nodes()
