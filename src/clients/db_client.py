from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from src.config import load_config

class DBClient:
    def __init__(self, url="http://localhost:6333", vec_size=1024):
        self.client = QdrantClient(url=url)
        self.vec_size = vec_size
        self.chunk_summaries = "chunk_summaries"
        self.entity_descriptions = "entity_descriptions"
        
        # To add
        #self.claim_summary = "claim_summary"
        #self.community_summary = "community_summary"

    def setup_collections(self):
        for label in [self.chunk_summaries, self.entity_descriptions]:
            if not self.client.collection_exists(label):
                self.client.create_collection(collection_name=label, vector_config=VectorParams(size=self.vec_size, distance=Distance.COSINE))
                print(f"Created new database collection for: {label}")

    def _gen_set_id(self, id_string):
        # Generate deterministic uuid for given string (wont reset and string not allowed for identifier)
        uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id_string)))
        
        return uuid
    
    def save_chunk(self, chunk_data):
        points = []

        for chunk in chunk_data.to_dict('records'):
            points.append(PointStruct(
                id= self._gen_set_id(chunk_data['id']),
                vector=chunk['embedding'].tolist(),
                payload={
                    "text_summary": chunk["text"], 
                    "doc_id": chunk.get('doc_id'),
                    "page": chunk.get('page'),
                    "bbox": chunk.get('bbox')
                }
            ))

        self.client.upsert(collection_name="chunk_summaries", points=points)

        return points
    
    def save_entity(self, entity_data):
        points = []

        for entity in entity_data.to_dict('records'):
            points.append(PointStruct(
                id = self._gen_set_id(entity['id']),
                vector= entity['embedding'].tolist(),
                payload={
                    "name": entity.get('name'),
                    "description": entity.get('description'),
                    "category": entity.get('category')
                }
            ))
        self.client.upsert(collection_name="entity_descriptions", points=points)

        return points
    
    def close(self):
        self.client.close()