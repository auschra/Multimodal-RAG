from src.config import load_config
import json
import requests
from src.retriever.hde import gen_hde
from qdrant_client import QdrantClient
from src.clients.embedding_client import EmbeddingClient # Your existing client

client = QdrantClient(url="http://localhost:6333")
embedder = EmbeddingClient(device='cpu')

def retrieve(query_text, collection_name, limit=5):
    emb_query = embedder.embed_text(query_text)
    
    # 3. Search Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=emb_query,
        limit=limit,
        with_payload=True
    )
    
    return results

query = "What is the lipschitz function?"
query_doc = gen_hde(query)

result = retrieve(query_doc, "chunk_summaries")
for p in result.points:
    print(p.score)
    print(p.payload['text_summary'])
