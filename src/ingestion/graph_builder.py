import os
from pathlib import Path
import networkx as nx
from src.ingestion.builder import BuildGraph
from src.config import config
import json


class BuildGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # Generate doc node -> doc_id = stem
    def create_doc_node(self, doc_stem):
        """
        Generate a document node and assigns its stem as id.

        Args: 
            doc_stem (str): document file stem        
        Returns: 
            document id
        """

        doc_node_id = f"doc:{doc_stem}"
        self.G.add_node(doc_node_id, type="Document", title=doc_stem)

        return doc_node_id
    
   
    def assign_node(self, doc_node_id, chunk):
        """
        Creates a chunk node, uses its chunk_id key for node.

        Args:
            doc_node_id: document id which chunk originated from
            chunk: chunk dict
        Returns:
            chunk node id
        """
        chunk_node_id = chunk["chunk_id"]
        block_metas = json.dumps(chunk.get("block_metas", []))                      # all block metas it contains into string
        
        self.G.add_node(
            chunk_node_id,
            type="chunk",
            page_content=chunk.get("text", ""),
            summary_text="",
            heading=chunk.get("heading", ""),
            token_count=chunk.get("token_count", 0),
            block_metas=block_metas,
        )

        self.G.add_edge(doc_node_id, chunk_node_id, relationship="CONTAINS")        # assign to parent doc

        return chunk_node_id
        
    # Add the chunk extraction data to graph
    # Entities in different chunk share node
    def semantics_to_graph(self, chunk_node_id, extraction):
        """
        Adds chunk extraction outputs to graph. ie, summaries, entities and relationships
        
        Args:
            chunk_node_id:
            extraction (dict): All extractions from extractor.py
        Returns:
            - creates the nodes for all necessary components of chunk (summary added to the chunk node)
            - creates necessary relationships between these components
        """

        # Create entity map to its id ('this entity' -> 'ent:this_entity')
        def get_entity_node_id(name_string):
            return f"ent:{name_string.strip().lower().replace(' ', '_')}"
        
        summary = extraction.get("chunk_summary", "").strip()        # add summary to existing chunk
        if summary:
            self.G.nodes[chunk_node_id]["summary"] = summary        

        for entity in extraction.get("entities", []):
            raw_name = entity.get("name", "").strip()       
            entity_id = get_entity_node_id(raw_name)

            # Add to graph if not existing
            if entity_id not in self.G:
                self.G.add_node(
                    entity_id,
                    type="entity",
                    label=raw_name,
                    category=entity.get("type", ""),
                    description=entity.get("description", ""),
                )

            else:
                # Hold both descriptions, use LLM later on to filter 
                old_desc = self.G.nodes[entity_id].get("description", "")
                new_desc = entity.get("description", "")
                if len(new_desc) > len(old_desc):
                    self.G.nodes[entity_id]["description"] = new_desc
            
            self.G.add_edge(chunk_node_id, entity_id, relation="MENTIONS")
        

        # Add relationships etween entities
        for rel in extraction.get("relationships", []):
            source_raw = rel.get("source", "")
            target_raw = rel.get("target", "")
            rel_type = rel.get("type", "").strip().upper()

            # Skip if either endpoint wasnt extracted as entity
            if not (source_raw and target_raw and rel_type):
                continue
            
            source_id = get_entity_node_id(source_raw)
            target_id = get_entity_node_id(target_raw)
            
            if source_id not in self.G:
                self.G.add_node(source_id, type="entity", label=source_raw.strip(), category="Unknown", description="")
            if target_id not in self.G:
                self.G.add_node(target_id, type="entity", label=target_raw.strip(), category="Unknown", description="")
    
            self.G.add_edge(source_id, target_id, relation=rel_type)
    


    def save(self, path):
        # save as json node link format
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G, edges="edges")              # NetworkX prefers edges key
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

    # Build graph
def main():
    builder = BuildGraph()


    # TODO change all these to use clean config dir
    graph_path = config.dirs.graph / "nodes.json" 
    entities_path = Path("/datasets/scratch/04_entities/entities.jsonl")        
    output_path = config.dirs.graph / "full_graph.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = Path("/datasets/scratch/02_chunks")

    # Build document and chunk nodes
    for json_file in chunk_dir.glob("*.json"):
        stem = json_file.stem.replace("_chunks", "")
        doc_node_id = builder.create_doc_node(stem)

        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            builder.assign_node(doc_node_id, chunk)



    # Extractions added to graph
    if entities_path.exists():
        with entities_path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                chunk_id = record["chunk_id"]
                
                if chunk_id in builder.G:
                    builder.semantics_to_graph(chunk_id, record["extraction"])

    else:
        print(f"Error -> metadata file {entities_path} missing")

    print(f"Graph built with: ({builder.G.number_of_nodes()} nodes. save to{output_path}")
    builder.save(output_path)

if __name__ == '__main__':
    main()