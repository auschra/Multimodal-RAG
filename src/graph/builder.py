import networkx as nx
import json
from pathlib import Path

class BuildGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # Generate doc node from stem name, assign unique deterministic id
    def create_doc_node(self, doc_stem):
        doc_node_id = f"doc:{doc_stem}"
        self.G.add_node(doc_node_id, type="Document", title=doc_stem)

        return doc_node_id
    
    # Assign chunk node, tie to paretn doc via contains rel
    def assign_node(self, doc_node_id, chunk):
        chunk_node_id = chunk["chunk_id"]
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
        
    # Add the chunk extraction data to graph
    # Entities in different chunk share node
    def semantics_to_graph(self, chunk_node_id, extraction):
        
        # Add chunk summary to existing chunk nodde
        summary = extraction.get("chunk_summary", "").strip()
        if summary:
            self.G.nodes[chunk_node_id]["summary"] = summary        
                    
        # Add entity (normalise name to preven dupes) (entity id should match entity name (lowercase))
        entity_map = {}
        for entity in extraction.get("entities", []):
            raw_name = entity.get("name", "").strip()
            lowercase_name = raw_name.lower()
            entity_id = f"ent:{lowercase_name.replace(' ', '_')}"
            entity_map[lowercase_name] = entity_id
            
            # check if entity exists, add if not
            if entity_id not in self.G:
                self.G.add_node(
                    entity_id,
                    type="entity",
                    label=raw_name,
                    category=entity.get("type", ""),
                    description=entity.get("description", ""),
                )

            else:
                # Pick the better description to use (just use longer one for now)
                old_desc = self.G.nodes[entity_id].get("description", "")
                new_desc = entity.get("description", "")

                if len(new_desc) > len(old_desc):
                    self.G.nodes[entity_id]["description"] = new_desc

            
            
            self.G.add_edge(chunk_node_id, entity_id, relation="MENTIONS")
        
        # entity-entity relationship edge
        for rel in extraction.get("relationships", []):
            source = rel.get("source", "").strip().lower()
            target = rel.get("target", "").strip().lower()
            rel_type = rel.get("type", "").strip().upper()

            if not (source and target and rel_type):
                continue
            
            source_id = entity_map.get(source)
            target_id = entity_map.get(target)
            
            # Skip if either endpoint wasn't extracted as an entity
            if source_id is None or target_id is None:
                continue
    
            self.G.add_edge(source_id, target_id, relation=rel_type)
    
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

