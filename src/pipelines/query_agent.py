import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config

from src.retriever.retriever import vis_retrieval
from src.generator.vlm_client import generate_answer

cfg = load_config()

def get_pdf_img(pdf_name, page_num):

    pdf_path = cfg.dirs.raw_data / f"{pdf_name}.pdf"
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=600)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

    return img

def get_bounding_box(response, width, height):
    # returns the pixel bounding box for the relevant information on the page from VLM

    pass

def main():
    query = "How many red dots do you see in each of the Lipschitz function visualisations?"    
    # Retrieve top-k restults -> List[(score, embedding.pt), ]
    print(f"Performing top k retreival")
    top_k_results = vis_retrieval(query, top_k=3)


    # List of retrieved PDF pages as image
    retrieved_images = []
    for score, ref in top_k_results:
        # Parent document, page number
        parts = ref.split("/")
        doc_name = parts[0]
        page_idx = int(parts[1].split("_")[1].split(".")[0]) 
        img = get_pdf_img(doc_name, page_idx)
        retrieved_images.append(img)

    print(f"No. images retrieved: {len(retrieved_images)}")

        
    # Append images as context for VLM text/image -> text generator 
    print("Generating answer")
    final_answer = generate_answer(query, retrieved_images)
    
    print(f"\nFinal Answer:\n{final_answer}")

    # display retrieved images with attention masking 


if __name__ == "__main__":
    main()