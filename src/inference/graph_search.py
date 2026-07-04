from src.config import config
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def graph_metrics(G):
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    density = nx.density(G)
    degree = G.degree()

    return (f"n_nodes: {n_nodes}, n_edges: {n_edges}, density: {density}, degree:  {degree}")

def traverse(query_embedding: list[float], 
             initial_chunks: list[dict], 
             embedding_client, 
             graph=G, 
             cs_threshold=0.7):
    """
    Receives list of retrieved chunks, performs multi step graph traversal based on 
    extracted chunk relationships
    Returns: updated list of all inclusive chunks
    """
    read_chunks = {chunk["chunk_id"] for chunk in initial_chunks}
    expanded = list(initial_chunks)

    for chunk in initial_chunks:
        # Get entities in this chunk
        entity_ids = [ent for ent in G.neighbors(chunk["chunk_id"]) if G.nodes[ent].get("type") == "entity"]

        # Embed entity description to see if its relevant to query
        for entity_id in entity_ids:
            entity_desc = G.nodes[entity_id].get("description", "")
            entity_embedding = embedding_client.embed_text(entity_desc)
            similarity = cosine_similarity(query_embedding, entity_embedding)

            # Filter out non-relevant entities
            if similarity < cs_threshold:
                continue
                
            # For relevant entities, find chunks they exist in, not already in read_chunks
            new_chunks = [node for node in G.neigbours(entity_id) if G.node.get("type") == "chunk"
                          and node not in read_chunks]

            # Add these chunks to read chunks, and expanded network
            for chunk_id in new_chunks:
                read_chunks.add(chunk_id)
                expanded.append({"chunk_id": chunk_id, "source": "graph_hop"})

    return expanded