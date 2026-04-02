import pymupdf
from pathlib import Path
import sys
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from src.config import load_config
from docling.chunking import HybridChunker

cfg = load_config()

# Load paths and doc
raw_path = cfg.dirs.raw_data
processed_path = cfg.dirs.processed_data 

# Docling setup
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True 

chunker = HybridChunker

# Ingest -> chunk text and pass pdf images to image_queue for gpu worker. 
def docling_worker(pdf_queue, image_queue):

    converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options,)})

    while True:
        pdf_path = pdf_queue.get()
        
        if pdf_path is None:
            pdf_queue.task_done()
            break
        try:
            result = converter.convert(pdf_path).document
            md_text = result.export_to_markdown()

            # Save markdown text to file 
            stem = Path(pdf_path).stem
            with open(f"{processed_path}/text/{stem}.md", "w") as f:
                f.write(md_text)

            # Render image for gpu worker :)
            doc = pymupdf.open(pdf_path)
            for i, page in enumerate(doc):
                
                # Save in temporary directory before being eaten by GPU queue
                pixels = page.get_pixmap(dpi=300)
                temp_img_path = f"data/processed/temp_images/{stem}_page_{i}.png"
                pixels.save(temp_img_path)

                # Pass temp image path to GPU queue
                image_queue.put((temp_img_path, stem, i))

        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

        # Queue finished
        finally:
            pdf_queue.task_done()
              



    