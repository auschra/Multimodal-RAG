from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

client = QdrantClient(path="~/datasets/qdrant_data/storage/")
vec_size = 1024 # BAAI/bge-m3 size

# Create new database collection for each abstraction level
def create_collections():
    for label in ["chunk_summaries", "entity_descriptions"]:        # List of summaries to create -> may add communities/sections
        if not client.collection_exists(label):
            client.create_collection(collection_name=label, vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),)

create_collections()

# Save data to its corresponding collection
def save_data(chunk_data, entity_data):

    chunk_points = []
    for chunk in chunk_data:
        chunk_points.append(PointStruct(
                                id=str(uuid.uuid4()),
                                vector=['embedding'],
                                payload={"text_summary": chunk["text_summary"],             # Full text
                                         "doc_id": chunk['doc_id'],                         # Parent doc node
                                         "entities": chunk['entities']}))                   # Contained entities
    client.upsert(collection_name="chunk_summaries", points=chunk_points)                   # Upload to collection

    ent_points = []
    for ent in entity_data:
        ent_points.append(PointStruct(
                                id=ent['label'],
                                vector=['embedding'],
                                payload={"description": ent['description'],                       # Full text
                                        # "chunk": ent['doc_id'],                                 # Parent chunk node
                                         "rel_entities": ent['rel_entities']}))                   # Connected entities
    client.upsert(collection_name="ent_descriptions", points=ent_points)                          # Upload to collection

