import pymupdf
from pathlib import Path
import sys
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available, get_torch_device
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
import torch

# add src path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config
cfg = load_config()

# Load paths and doc
raw_path = cfg.dirs.raw_data / "geom.pdf"
processed_path = cfg.dirs.processed_data


"""TODO
- implement parallel page ingestion and processing
- parallel saturation of gpu for embedding"""

# Take pymupdf page and output PIL image for colpali
def ingest_single_page(page, dpi=200):
    pixels = page.get_pixmap(dpi=dpi)
    mode = "RGB"
    image = Image.frombytes(mode, [pixels.width, pixels.height], pixels.samples)
    
    return image

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)
query = "Lipschitz function"
q_in = processor.process_queries([query]).to(model.device)

with torch.no_grad():
        q_embeddings = model(**q_in)
save_file = processed_path / f"query.pt"

doc = pymupdf.open(raw_path)

for i, page in enumerate(doc):
    image = ingest_single_page(page)
    batch_inputs = processor.process_images([image]).to(model.device)

    with torch.no_grad():
        embeddings = model(**batch_inputs)

    save_file = processed_path / f"page_{i}_emb.pt"
    torch.save(embeddings, save_file)


