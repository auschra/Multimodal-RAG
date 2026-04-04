import networkx as nx
import uuid


# Document structure
class GraphGenerator:
    def __init__(self):
        self.G = nx.DiGraph

    # Create parent doc node
    def intake_text(self, title, metadata):

        doc_node_id = f"doc_{uuid.uuid4().hex[:8]}"
        self.G.add_node(doc_node_id, type="Document", title=title, **metadata)                          # Create doc node

        return doc_node_id
    
    # Create child chunk node and assign to parent doc
    def assign_node(self, doc_node_id, chunk_text, vec_id, page_no):

        chunk_node_id = f"chunk_{uuid.uuid4().hex[:8]}"
        self.G.add_node(chunk_node_id, type="Chunk", text=chunk_text, vec_id=vec_id, page=page_no)      # Create chunk node
        self.G.add_edge(doc_node_id, chunk_node_id, relationshup="CONTAINS")                            # Assign to doc

        return chunk_node_id
