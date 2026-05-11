import json
from pathlib import Path
from typing import Iterator
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from src.graph.builder import BuildGraph
import uuid

# All data present in graph for embdding (full chunk text, chunk summaries, entities (also rels etc))
graph_path = Path("/datasets/scratch/05_graph/full_graph.json")
model_name = "BAAI/bge-m3"
embed_size = 1024
batch_size = 32

# Map entity names to deterministic uuid for qdrant
def str_to_uuid(string_id):
    uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))
    
    return uuid

# Get the specific data type for collection creation (chunk, summaries or entities) from graph
def all_jobs(graph):

    for node_id, data in graph.G.nodes(data=True):
        node_type = data["type"]
        
        if node_type == "chunk":
            text = data["text"]
            heading = data["heading"]
            
            # Separate collections for chunk summaries and full chunk text embedding
            if text:
                yield "chunks", {
                    "id": node_id,                               # already uuid
                    "embedded": f"[{heading}]\n{text}" if heading else text,
                    "payload": {
                        "type": "chunk",
                        "node_id": node_id,
                        "heading": heading,
                        "snippet": text[:200],
                    },
                }
            
            summary = data["summary"]

            if summary:
                yield "summaries", {
                    "id": node_id,                                  # already uuid
                    "embedded": summary,
                    "payload": {
                        "type": "summary",
                        "node_id": node_id,
                        "heading": heading,
                        "summary_text": summary,
                    },
                }
        
        elif node_type == "entity":
            label = data["label"]
            desc = data["descriptions"]

            if label and desc:
                yield "entities", {
                    "id": str_to_uuid(node_id),             # convert ent name to uuid
                    "embedded": f"{label}: {desc}",
                    "payload": {
                        "type": "entity",
                        "node_id": node_id,                 # keep original id in payload
                        "label": label,
                        "category": data["category"],
                        "description": desc,
                    },
                }

# Check if collection exists, create new if not
def ensure_collection(client, name, dim):

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),)

# embed text, add as point to collection
def embed_and_upsert(model, client, collection, points):

    # batch encode batch amount per collection 
    for i in range(0, len(points), batch_size):

        batch = points[i:i + batch_size]
        texts = [p["text_to_embed"] for p in batch]
        vectors = model.encode(texts, convert_to_numpy=True).tolist()   
        
        # Create qdrant point out of emb vec and payload
        q_points = [PointStruct(id=p["id"], vector=v, payload=p["payload"]) for p, v in zip(batch, vectors)]

        # send to collection
        client.upsert(collection_name=collection, points=q_points)
        
def main():
    # Setup embedder and qdrant client
    model = SentenceTransformer(model_name, device="cuda")
    client = QdrantClient(host="localhost", port=6333)

    # Check for existing or create new collections
    for name in ("chunks", "summaries", "entities"):
        ensure_collection(client, name, embed_size)

    graph = BuildGraph.load(graph_path)
    
    #  all points by destination collection
    all_collections = {"chunks": [], "summaries": [], "entities": []}

    # Add vec, payload to corresponding collection for staging
    for collection, point in all_jobs(graph):
        all_collections[collection].append(point)

    # execute embedding for collection lists and send to qdrant
    for collection, points in all_collections.items():
        embed_and_upsert(model, client, collection, points)

if __name__ == "__main__":
    main()