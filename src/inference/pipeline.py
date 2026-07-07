from typing import Dict, Any, List
import networkx as nx
from openai import OpenAI
from src.config import config

from src.clients import db_client
from src.clients import embedding_client
from src.clients import vlm_client
from src.clients import colpali_client

from src.ingestion.graph_builder import BuildGraph

from src.inference.router import route_query
from src.inference.graph_search import traverse
from src.inference.vector_search import retrieve_chunks
from src.inference.generator import generate

class InferencePipeline:
    def __init__(self, load_colpali=False):                     # lazy load colpali due to vram
        self.llm_client = OpenAI(
            base_url = config.network.vllm_api_base,
            api_key = "EMPTY",
        )

        self.llm_model = config.models.llm_model

        # Setup clients
        self.qdrant_client = db_client.QdrantClient()
        self.embedding_client = embedding_client.EmbeddingClient()
        self.llm_client = vlm_client.LLMClient()
        self.colpali_client = colpali_client.CPClient()

        self.builder = BuildGraph.load(config.dirs.graph / "full_graph.json")
        self.G = self.builder.G

    def run(self, query):

        # Route query to get retrieval mode (text, visual, hybrid)
        retrieval_mode = route_query(query, self.llm_client)

        # all methods require query embedding
        query_embedding = self.embedding_client.embed_text(query)

        # Handle hybrid and visual as the same for now
        if retrieval_mode == "visual" or retrieval_mode == "hybrid":
            pass

            
        # Always pass to text retrieval
        chunks = retrieve_chunks(query_embedding, self.qdrant_client, collection="chunk_summaries", top_k=5)
        # Multi-step graph traversal -> chunk_ids of relevant chunks
        hopped_chunks = traverse(query_embedding, 
                                    initial_chunks=chunks,
                                    embedding_client=self.embedding_client,
                                    graph=self.G,
                                    cs_threshold=0.5)
        
        answer = generate(query, hopped_chunks, self.llm_client)

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