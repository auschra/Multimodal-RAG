from typing import Dict, Any, List
import networkx as nx
from openai import OpenAI
from src.config import config
from src.clients import db_client
from src.clients import embedding_client
from src.clients import vlm_client

from src.inference.router import route_query
from src.inference.graph_search import traverse
from src.inference.vector_search import retrieve_chunks
from src.inference.generator import generate

class InferencePipeline:
    def __init__(self, load_colpali=False):                     # lazy load colpali due to vram
        self.llm_client = OpenAI(
            base_url = config.vllm_api_base,
            api_key = "EMPTY",
        )

        self.vlm_model = config.models.vlm_model

        # Setup clients
        self.qdrant_client = db_client.QdrantClient()
        self.embedding_client = embedding_client.EmbeddingClient()
        self.llm_client = vlm_client.LLMClient()

        self.G = nx.read_graphml(config.dirs.graph / "knowledge_graph.graphml")

    def run(self, query):

        # Route query to get retrieval mode (text, visual, hybrid)
        retrieval_mode = route_query(query, self.vlm_client)

        # all methods require query embedding
        query_embedding = self.embedding_client.embed_text(query)


        if retrieval_mode == 'text':
            
            # Initial chunk retrieval
            chunks = retrieve_chunks(query_embedding, self.qdrant_client, collection="chunks", top_k=5)

            # Multi-step graph traversal -> chunk_ids of relevant chunks
            hopped_chunks = traverse(query_embedding, 
                                     initial_chunks=chunks,
                                     embedding_client=self.embedding_client,
                                     graph=self.G,
                                     cs_threshold=0.7)
            
            answer = generate(query_embedding, hopped_chunks, self.llm_client)

            return answer

        if retrieval_mode == 'visual':
            pass

        if retrieval_mode == 'hybrid':
            pass

        
        # Embed query
        

        # Top-k sim chunks
        

        return None

   
    
    def colpali_processing(self, all_chunks):
        # take top relevant chunks, get pages out and embed with colplai

        # extract information from pages + chunk text to generate answer
        page_vis_embeddings = all_chunks

        return page_vis_embeddings