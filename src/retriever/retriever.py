import sys
from pathlib import Path
# remove when running from root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import pymupdf
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from src.config import load_config

cfg = load_config()


# Colpali setup
print(f"Loading retriever model ({cfg.models.colpali_model})")
model = ColQwen2.from_pretrained(
    cfg.models.colpali_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,).eval()
processor = ColQwen2Processor.from_pretrained(cfg.models.colpali_model)
print("Retriever model loaded")

def vis_retrieval(query, top_k=None):
    top_k = top_k or cfg.retrieval.top_k
    embeddings_path = cfg.dirs.embeddings    

    # ------- for now, load query here and embed -------
    q_in = processor.process_queries([query]).to(model.device)
    # Embed query
    with torch.no_grad():
        q_embeddings = model(**q_in)

    # run late interaction between query and each embedding 
    # load embeddings for document

    # ----------------------

    # keep track of (score, page) for each embedding
    all_scores = []

    # Iterate through each document's embeddings
    for doc_folder in embeddings_path.iterdir():
        if not doc_folder.is_dir():
            continue

        print(f"Loading embeddings: {doc_folder.name}")
        
        page_embeddings = []
        page_references = []
        
        """
        # Load embeddings.pt in memory on device
        for pt_file in doc_folder.glob("*.pt"):

            emb = torch.load(pt_file, map_location=model.device, weights_only=True)
            page_embeddings.append(emb)
            page_references.append(f"{doc_folder.name}/{pt_file.name}")
        """


        # ---------- temporary ----------
        for pt_file in doc_folder.glob("*.pt"):
            # Load the tensor directly onto the GPU
            emb = torch.load(pt_file, map_location=model.device, weights_only=True)
            
            # FIX: Squeeze out the batch dimension so (1, N, D) becomes (N, D)
            if emb.dim() == 3 and emb.size(0) == 1:
                emb = emb.squeeze(0)
                
            page_embeddings.append(emb)
            page_references.append(f"{doc_folder.name}/{pt_file.name}")
        # ---------------------------

        if not page_embeddings:
            continue

        # Multi-vec MaxSim 
        with torch.no_grad():
            # score_multi_vector: expects query embedding, list of document page embeddings
            scores = processor.score_multi_vector(q_embeddings, page_embeddings)
            
        # scores = (1, page_ref)
        # Pair score with corresponding page
        for score, ref in zip(scores[0], page_references):
            all_scores.append((score.item(), ref))

   # Sort scores from all documents
    all_scores.sort(key=lambda x: x[0], reverse=True)

    final_results = all_scores[:top_k]

    # for debug
    for i, (score, ref) in enumerate(final_results):
        print(f"{i+1}: Score: {score:.4f} | Page: {ref}")

    return final_results

if __name__ == "__main__":
    test_query = "How many red dots are there in the third figure of the Lipschitz function?"
    vis_retrieval(test_query)