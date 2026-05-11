import sys
import logging
from qdrant_client import QdrantClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import argparse 

# demo of querying database

def main():
    query = 'How was gpt3 able to handle more complex tasks compared with gpt 2. What architectural or emergent capability was responsible?'
    
    qdrant = QdrantClient(host="localhost", port=6333)
    llm = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
    embedder = SentenceTransformer("BAAI/bge-m3")
    
    print(f"using query {query}")
    print('embedding query')
    query_vec = list(embedder.encode([query]))[0].tolist()
    
    print('performing vector search')
    results = qdrant.query_points(
        collection_name="chunks",
        query=query_vec,
        limit=5,
        with_payload=True,
    )
    for point in results.points:
        print(f"ID: {point.id}, Score: {point.score}, Heading: {point.payload.get('heading')}, Content: {point.payload.get('snippet')}")


if __name__ == "__main__":
    main()