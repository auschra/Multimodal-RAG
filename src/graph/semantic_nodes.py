from pathlib import Path
import json
from src.graph.builder import BuildGraph

def main():
    # hardcode these for now, switch to relative later
    graph_path = Path("/datasets/scratch/03_nodes/nodes.json")    # chunk and doc nodes
    entities_path = Path("/datasets/scratch/04_entities/entities.jsonl")
    output_path = Path("/datasets/scratch/05_graph/full_graph.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder = BuildGraph.load(graph_path)
    print(f"Loaded graph: {builder.G.number_of_nodes()} nodes, {builder.G.number_of_edges()} edges")

    chunk_ids_in_graph = {n for n, attrs in builder.G.nodes(data=True) if attrs.get("type") == "chunk"}
    print(f"Graph has {len(chunk_ids_in_graph)} chunk nodes")
    print(f"Sample chunk IDs in graph: {list(chunk_ids_in_graph)[:3]}")

    n_chunks = 0
    n_skipped = 0
    
    with entities_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_id = record["chunk_id"]
            
            if chunk_id not in builder.G:
                if n_skipped < 3:
                    print(f"Skipped chunk_id (not in graph): {chunk_id}")
                n_skipped += 1
                continue
            
            builder.semantics_to_graph(chunk_id, record["extraction"])
            n_chunks += 1
    
    builder.save(output_path)

if __name__ == "__main__":
    main()