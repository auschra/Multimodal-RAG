import pymupdf
from pathlib import Path
import sys
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,)
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
import torch

# add src path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config
cfg = load_config()

# Load paths and doc
raw_path = cfg.dirs.raw_data
processed_path = cfg.dirs.processed_data 
embeddings_path = cfg.dirs.embeddings

# Take pymupdf page and output PIL image for colpali
def ingest_single_page(page, dpi=200):

    pixels = page.get_pixmap(dpi=dpi)
    mode = "RGB"
    image = Image.frombytes(mode, [pixels.width, pixels.height], pixels.samples)
    
    return image

# Colpali setup
model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

# Iterate through PDFs in data/raw
for path in raw_path.iterdir():
    if path.is_file():
        print(f"File found: {path.name}")

        # Create embeddings folder for file
        file_embed_path = Path(embeddings_path / path.stem)
        file_embed_path.mkdir(parents=True, exist_ok=True)
        print(f"Embeddings folder create {file_embed_path}")

        # Create data/processed folder for file
        file_processed_path = Path(processed_path / path.stem)
        file_processed_path.mkdir(parents=True, exist_ok=True)
        print(f"Processed fodler created: {file_processed_path}")

        # Open file with PymuPDF
        doc = pymupdf.open(path)

        # Generate page embeddings, save to embeddings folder
        for i, page in enumerate(doc):
            
            image = ingest_single_page(page)
            batch_inputs = processor.process_images([image]).to(model.device)

            with torch.no_grad():
                embeddings = model(**batch_inputs)
            page_embed_file = file_embed_path / f"page_{i}_emb.pt"
            torch.save(embeddings, page_embed_file)

            # Text processing TODO
            text = None

    elif path.is_dir():
        print(f"Dir found: {path}")
