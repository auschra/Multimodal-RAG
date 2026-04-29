import networkx as nx
import json
from pathlib import Path

class BuildGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # Generate doc node from stem name, assign unique deterministic id
    def create_doc_node(self, doc_stem):

        doc_node_id = f"doc:{doc_stem}"                                                                         # string doc id for networkx
        self.G.add_node(doc_node_id, type="Document", title=doc_stem)                                           # Create doc node

        return doc_node_id
    
    def assign_node(self, doc_node_id, chunk):
        chunk_node_id = chunk["chunk_id"]
        
        # graphml needs json string
        block_metas = json.dumps(chunk.get("block_metas", []))
        
        self.G.add_node(
            chunk_node_id,
            type="chunk",
            text=chunk.get("text", ""),
            heading=chunk.get("heading", ""),
            token_count=chunk.get("token_count", 0),
            block_metas=block_metas,
            # new fields
        )

        self.G.add_edge(doc_node_id, chunk_node_id, relationship="CONTAINS")
        return chunk_node_id
    
    def save(self, path):
        # save as json node link format
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G, edges="edges")              # newer NetworkX prefers explicit edges key
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
       # recreate graph some save file
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        builder = cls()
        builder.G = nx.node_link_graph(data, edges="edges", directed=True)
        return builder

