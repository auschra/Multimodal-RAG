from pathlib import Path
from typing import Iterator
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from src.ingestion.graph_builder import BuildGraph
import uuid
from src.config import config

# All data present in graph for embdding (full chunk text, chunk summaries, entities (also rels etc))
graph_path = config.dirs.graph / "full_graph.json"
model_name = config.models.embedding_model
embed_size = config.models.embed_dim
batch_size = config.models.emb_bs

CHUNK_COLLECTION = "chunk_summaries"
ENTITY_COLLECTION = "entity_descriptions"

# Map entity names to deterministic uuid for qdrant
def str_to_uuid(string_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))

# Get the specific data type for collection creation (chunk, summaries or entities) from graph
def all_jobs(graph):
    """
    
    """

    for node_id, data in graph.G.nodes(data=True):
        node_type = data["type"]
        
        if node_type == "chunk":
            text = data.get("page_content", "")
            heading = data.get("heading", "")
            
            # Separate collections for chunk summaries and full chunk text embedding
            if text:
                yield CHUNK_COLLECTION, {
                    "id": node_id,                               # already uuid
                    "embedded": f"[{heading}]\n{text}" if heading else text,
                    "payload": {
                        "type": "chunk",
                        "chunk_id": node_id,
                        "heading": heading,
                        "page_content": text,
                        "snippet": text[:200],
                    },
                }
            summary = data.get("summary_text")

            if summary:
                yield CHUNK_COLLECTION, {
                    "id": node_id,                                  # already uuid
                    "embedded": summary,
                    "payload": {
                        "type": "summary",
                        "chunk_id": node_id,
                        "heading": heading,
                        "summary_text": summary,
                        "page_content": text,       # fallback to raw text
                    },
                }
        
        elif node_type == "entity":
            label = data.get("labe", "")
            desc = data.get("description", "")

            if label and desc:
                yield ENTITY_COLLECTION, {
                    "id": str_to_uuid(node_id),             # convert ent name to uuid
                    "embedded": f"{label}: {desc}",
                    "payload": {
                        "type": "entity",
                        "entity_id": node_id,                 # keep original id in payload
                        "label": label,
                        "category": data.get("category", ""),
                        "description": desc,
                    },
                }

def ensure_collection(client, name, dim):
    """
    Check collection exists, create if not.
        
    Args:
        client (obj): embedding client
        name (str): collection name
        dim (int): embedding_dim
    Returns:
        creates the collection
    """

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),)

def embed_and_upsert(model, client, collection, points):
    """
    Embeds text, adds it as point to collection.
    Args:
        model: 
        client:
        collection
        points:
    Returns:

    """

    # batch encode batch amount per collection 
    for i in range(0, len(points), batch_size):

        batch = points[i:i + batch_size]
        texts = [p["embedded"] for p in batch]
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
    for name in (CHUNK_COLLECTION, ENTITY_COLLECTION):
        ensure_collection(client, name, embed_size)

    graph = BuildGraph.load(graph_path)
    #  all points by destination collection
    all_collections = {CHUNK_COLLECTION: [], ENTITY_COLLECTION: []}

    # Add vec, payload to corresponding collection for staging
    for collection, point in all_jobs(graph):
        all_collections[collection].append(point)

    # execute embedding for collection lists and send to qdrant
    for collection, points in all_collections.items():
        embed_and_upsert(model, client, collection, points)

if __name__ == "__main__":
    main()