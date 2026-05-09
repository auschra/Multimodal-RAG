from pathlib import Path
import json
from src.graph.builder import BuildGraph

"""
Add extracted entities and relationships to graph.
"""

entities_path = Path("/datasets/scratch/04_entities/entities.jsonl")

with open(entities_path, "r", encoding='utf-8') as f:
    chunks = 0
    total_ents = 0
    for line in f:
        chunk_extract = json.loads(line)
        chunks += 1 

        entities = [ent for ent in chunk_extract['extraction']['entities']]
        relationships = [rel for rel in chunk_extract['extraction']['relationships']]
        n_ents = len(entities)
        print(n_ents)
        total_ents += n_ents

        for single_ent in entities:
            name = entities['name']
            ent_type = entities['type']
            description = entities['description']

        for rel in relationships:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']

print(len(chunk_extract))
print(f"N chunkks {chunks}")
"""
graph = BuildGraph.load()

for ent in entities:
    extracted = ent

    for c in extracted:
        graph.assign_node(c)

        # relationship info


        # entity format
        #{chunk_id, extraction: {chunk_summary, entities: [{name, type, description}], relationships: [{source, target, type}]}}

"""
"""

{"chunk_id": "f9743f65-5802-4d5c-b922-027a455b5d79", 
"extraction": {"chunk_summary": "The text discusses improvements made for competitions using deep features and residual learning.", 
                "entities": [{"name": "CompetitionImprovement", "type": "Metric", "description": "Improvements made for competitions based on deep features."}, 
                            {"name": "DeepFeatures", "type": "Dataset", "description": "Features used in deep learning models to enhance competition performance."}, 
                            {"name": "ResidualLearning", "type": "Algorithm", "description": "Learning method that helps improve deep feature-based competition improvements."}], 
                "relationships": [{"source": "CompetitionImprovement", "target": "DeepFeatures", "type": "Uses"}, 
                                    {"source": "CompetitionImprovement", "target": "ResidualLearning", "type": "EnhancedBy"}]}}
"""