import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from src.config import config

def graph_metrics(graph):
    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)
    return (f"n_nodes: {n_nodes}, n_edges: {n_edges}")

def traverse(query_embedding: list[float], 
             initial_chunks: list[dict], 
             embedding_client, 
             graph, 
             cs_threshold:float = None)->list[dict]:
    """
    Receives list of retrieved chunks, performs multi step graph traversal based on 
    extracted chunk relationships
    Returns: updated list of all inclusive chunks
    """
    # tune later
    cs_threshold = cs_threshold or config.cs_threshold

    # Reshape query_embedding to 2d for sklearn cosine embd
    query_emb_2d = np.array(query_embedding).reshape(1, -1)

    read_chunks = {chunk["chunk_id"] for chunk in initial_chunks}
    expanded = list(initial_chunks)

    for chunk in initial_chunks:
        # Get entities in this chunk
        entity_ids = [ent for ent in graph.neighbors(chunk["chunk_id"]) if graph.nodes[ent].get("type") == "entity"]

        # Embed entity description to see if its relevant to query
        for entity_id in entity_ids:
            entity_desc = graph.nodes[entity_id].get("description", "")
            if not entity_desc:
                continue

            entity_embedding = embedding_client.embed_text(entity_desc)
            entity_emb_2d = np.array(entity_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_emb_2d, entity_emb_2d)[0][0]

            # Filter out non-relevant entities
            if similarity < cs_threshold:
                continue
                
            # For relevant entities, find chunks they exist in, not already in read_chunks
            new_chunk_ids = [node for node in graph.neighbours(entity_id) if graph.nodes[node].get("type") == "chunk"
                          and node not in read_chunks]

            # Add these chunks to read chunks, and expanded network
            for chunk_id in new_chunk_ids:
                read_chunks.add(chunk_id)
                node_data = graph.node[chunk_id]
                expanded.append({
                    "chunk_id": chunk_id,
                    "text": node_data.get("text", ""),
                    "document_id": node_data.get("document_id", ""),
                    "heading": node_data.get("heading", ""),
                    "score": float(similarity),   # use as a score
                    "source": "graph_hop",
                })

    return expanded