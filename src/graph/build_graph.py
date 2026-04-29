import os
from pathlib import Path
import networkx as nx
from src.graph.builder import BuildGraph
from src.config import load_config
import json
import concurrent.futures

def build_graph():
    builder = BuildGraph()
    print("Building graph.")

    tasks = []
    chunk_dir = Path(f"/datasets/scratch/02_chunks")

    # Iter through chunk files
    for json_file in chunk_dir.glob("*.json"):
        stem = json_file.stem.replace("_chunks", "")
        doc_node_id = stem                                                                                  # Create doc node from json file

        with open(json_file, "r", encoding="utf-8") as f:                                                   # Get all chunk objects in json file
            chunks = json.load(f)

        for chunk in chunks:                                                                                # Get chunk metadata and create node
            node = builder.assign_node(doc_node_id, chunk)

    # save 
    save_dir = Path("/datasets/scratch/03_nodes")
    save_dir.mkdir(parents=True, exist_ok=True)
    builder.save(f"{save_dir}/nodes.json")
            
if __name__ == '__main__':
    build_graph()
    

