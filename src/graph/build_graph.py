import os
from pathlib import Path
import networkx as nx
from src.graph.build import BuildGraph, extract_entities, add_semantics_to_graph
from src.config import load_config
import json
import concurrent.futures

cfg = load_config()

processed_dir = cfg.dirs.processed_data
text_dir = Path(f"{processed_dir}/text")
json_dir = Path(f"{processed_dir}/json")
embedding_dir = cfg.dirs.embeddings

def process_extraction_task(task):
    chunk_node_id, chunk_text, stem, chunk_id = task
    _, _, clean_chunk_id = chunk_id.partition("chunk_")
    print(f"Doc: {stem}, chunk: {clean_chunk_id}")
    extraction_data = extract_entities(chunk_text)
    
    return chunk_node_id, extraction_data

def build_graph():
    builder = BuildGraph()
    print("Building graph.")

    tasks = []

    # Main node creation (docs and chunks) -------------------
    for json_file in json_dir.glob("*_chunks.json"):

        stem = json_file.stem.replace("_chunks", "")
        doc_node_id = builder.intake_text(title=stem, metadata={"filepath": str(json_file)})                # Create doc node from json file

        with open(json_file, "r", encoding="utf-8") as f:                                                   # Get all chunk objects in json file
            chunks = json.load(f)

        for chunk in chunks:                                                                                # Get chunk metadata and create node
            chunk_text = chunk.get("text", "")
            if not chunk_text.strip():
                continue

            page_num = chunk.get("page_number", 0)
            
            vector_path = embedding_dir / f"{stem}_page_{page_num}.pt"
            vector_id = str(vector_path) if vector_path.exists() else None
            
            chunk_node_id = builder.assign_node(
                doc_node_id=doc_node_id,
                chunk_text=chunk_text,
                vec_id=vector_id,
                page_no=page_num,
                chunk_type=chunk.get("chunk_type", "text"),
                bbox=chunk.get("bbox", []),
                heading=chunk.get("parent_heading", "")
            )
            
            
            tasks.append((chunk_node_id, chunk_text, stem, chunk.get("chunk_id")))                                  # Queue this as a task

    # Entity node and relationshup creation ------------------
    MAX_CONCURRENT_CHUNKS = 10
    print(f"\nStarting entity and relationship creation.")
    print(f"Total chunks: {len(tasks)}")
    
    results = []
    
    # Concurrency ------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS) as executor:
        assignments = [executor.submit(process_extraction_task, task) for task in tasks]

        for future in concurrent.futures.as_completed(assignments):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Chunk extraction generated an exception: {exc}")
    # EDges -------------------------
    print("\nConnecting edges")
    for chunk_node_id, extraction_data in results:
        add_semantics_to_graph(builder, chunk_node_id, extraction_data)

    # Save --------------------------
    output_path = Path(processed_dir) / "knowledge_graph.graphml"
    nx.write_graphml(builder.G, str(output_path))
    
    print(f"Total nodes: {builder.G.number_of_nodes()}")
    print(f"Total edges: {builder.G.number_of_edges()}")

if __name__ == "__main__":
    build_graph()