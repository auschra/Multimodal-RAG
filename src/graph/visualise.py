import networkx as nx
from pyvis.network import Network
from pathlib import Path
import html
from src.config import load_config

cfg = load_config()

def visualize_graph():
    graph_path = Path(cfg.dirs.processed_data) / "knowledge_graph.graphml"
    
    if not graph_path.exists():
        print(f"Graph not found at {graph_path}")
        return

    print("Loading graph into memory...")
    G = nx.read_graphml(str(graph_path))

    # Clean up node attributes for PyVis
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown")
        
        # Color coding by node type
        if node_type == "Document":
            data["color"] = "#4285F4"  # Google Blue
            data["size"] = 30
        elif node_type == "Chunk":
            data["color"] = "#34A853"  # Green
            data["size"] = 15
        else:
            data["color"] = "#EA4335"  # Red (Semantic Concepts)
            data["size"] = 20
            
        data["label"] = data.get("title", data.get("name", data.get("heading", str(node))))
        
        # FIX 2: Sanitize the hover text so it doesn't break the HTML/JS engine
        raw_text = str(data).replace("\n", " ").replace("\r", "")
        data["title"] = html.escape(raw_text)

    print("Generating interactive HTML...")
    
    # FIX 1: cdn_resources="remote" ensures it works when downloaded to your local PC
    net = Network(
        height="1000px", 
        width="100%", 
        bgcolor="#1a1a1a", 
        font_color="white", 
        select_menu=True,
        cdn_resources="remote" 
    )
    
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)
    net.from_nx(G)
    net.toggle_physics(True)
    
    output_file = "graph_visualization.html"
    net.write_html(output_file)
    print(f"Done. Download and open {output_file} in your web browser.")

if __name__ == "__main__":
    visualize_graph()