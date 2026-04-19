import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class EmbeddingClient():
    def __init__(self, model_name= 'BAAI/bge-m3', device='cpu'):  
        self.encoder = SentenceTransformer(model_name, device=device)                # Save VRAM (check speed diff)

    # Embed batch of strings -> return list of vectors
    def embed_text(self, texts, batch_size=12):
        embeddings = self.encoder.encode(texts, batch_size = batch_size)
        
        return embeddings.tolist()