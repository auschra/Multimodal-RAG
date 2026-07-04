from src.config import config
from src.clients.db_client import QdrantClient

def retrieve_chunks(query_embedding: list[float], 
                    qdrant: QdrantClient, 
                    collection: str = "chunks", 
                    top_k=5) -> list[dict]: 
    """
    Perform pure cosine similarity matching with embedded qdrant database
    Return top k chunks with metadata as initial nodes in graph."""


    results = qdrant.client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    chunks = [
        {   "chunk_id": r.id,
            "score": r.score,
            "text": r.payload.get("text", ""),
            "document_id": r.payload.get("document_id", ""),
            "heading": r.payload.get("heading", ""),
        } for r in results.points
    ]

    return chunks
