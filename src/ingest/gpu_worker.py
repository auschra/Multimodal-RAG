import pymupdf
from pathlib import Path
import sys
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
import torch
from src.config import load_config

cfg = load_config()

# Load paths and doc
raw_path = cfg.dirs.raw_data
processed_path = cfg.dirs.processed_data 
embeddings_path = cfg.dirs.embeddings

# Colpali setup
model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,).eval()
processor = ColQwen2Processor.from_pretrained(model_name)



def colpali_worker(image_queue, batch_size = 4):

    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,).eval()
    processor = ColQwen2Processor.from_pretrained(model_name)

    batch_paths = []
    batch_metadata = []

    while True:
        try:
            item = image_queue.get(timeout=3)

            if item is None:
                image_queue.task_done()
                break

            batch_paths.append(item[0])
            batch_metadata.append(item[1], item[2]) # stem, page number

        except image_queue.Empty: # partial batch
            pass

        # Execute when batch full or queue empty
        if len(batch_paths) >= batch_size or (image_queue.empty() and len(batch_paths) > 0):
            
            images = [Image.open(p) for p in batch_paths]
            batch_inputs = processor.process_images([images]).to(model.device)
            with torch.no_grad():
                embeddings = model(**batch_inputs)

            for i, (stem, page_num) in enumerate(batch_metadata):
                torch.save(embeddings[i], f"embeddings_path{stem}_page_{page_num}.pt")

                # Delete temporary image path
                Path(batch_paths[i]).unlink()

                image_queue.task_done()

            batch_paths = []
            batch_metadata = []
