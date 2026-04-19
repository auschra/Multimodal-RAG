import pymupdf
from pathlib import Path
import json
from collections import defaultdict
from transformers import AutoTokenizer
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from src.config import load_config

cfg = load_config()

# Load paths
raw_path = cfg.dirs.raw_data
processed_path = cfg.dirs.processed_data 

def docling_worker(pdf_queue, image_queue):
    """
    CPU tasks
    - parse PDF
    - Extract text & structural tags
    - standardise the coord system for chunk bounding boxes
    - group multiple boxes into per page provenance
    """

    # Docling options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True 

    # Setup objects
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)})
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512)
    
    while True:
        # Work through each assigned path in queue
        pdf_path = pdf_queue.get()                                                                          
        if pdf_path is None:
            pdf_queue.task_done()
            break

        stem = Path(pdf_path).stem    

        try:
            
            # Convert to docling document & also markdown file
            result = converter.convert(pdf_path).document                                                 
            md_text = result.export_to_markdown()                           

            chunk_list = []

            # Chunking through each doc
            for i, chunk in enumerate(chunker.chunk(result)):
                enriched_text = chunker.contextualize(chunk)                                                # Add brief heading to beiginning of chunk text
                heading = chunk.meta.headings[0] if chunk.meta.headings else "Root"                               
                
                chunk_type = "text"
                parent = []
                page_bboxes = defaultdict(list)                                                             # For chunks over >1 p page 

                # Fix for docling mixed coords to normalised top-left origin
                def get_norm_bbox(b, p_num):                       
                    page_height = result.pages[p_num].size.height                                              
                    origin = getattr(b, 'coord_origin', None)

                    if origin and str(origin).split('.')[-1] == "BOTTOMLEFT":
                        t_norm = page_height - b.top
                        b_norm = page_height - b.bot
                        return [b.left, min(t_norm, b_norm), b.right, max(t_norm, b_norm)]
                    return [b.left, b.top, b.right, b.bot]
                
                # Chunk containing heading
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:                                   
                    for head_node in chunk.meta.headings:
                        if hasattr(head_node, 'prov') and head_node.prov:
                            for prov in head_node.prov:
                                if prov.bbox:
                                    page_bboxes[prov.page_no].append(get_norm_bbox(prov.bbox, prov.page_no))    

                # Main body text
                if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    first_item_label = chunk.meta.doc_items[0].label
                    chunk_type = str(first_item_label).split('.')[-1].lower()
                    
                    for item in chunk.meta.doc_items:
                        item_label = item.label
                        
                        # Filter out extra text from chunks
                        noise_labels = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}
                        if item_label in noise_labels:
                            continue
                                                    
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if prov.bbox:
                                    page_bboxes[prov.page_no].append(get_norm_bbox(prov.bbox, prov.page_no))
                    
                for p_no, bboxes in page_bboxes.items():                                                        # Merge bboxes
                    if bboxes:
                        merged_bbox = [
                            round(min(b[0] for b in bboxes), 2),                    # Left
                            round(min(b[1] for b in bboxes), 2),                    # Top 
                            round(max(b[2] for b in bboxes), 2),                    # Right
                            round(max(b[3] for b in bboxes), 2)                     # Bottom
                        ]
                        parent.append({"page_number": p_no,"bbox": merged_bbox})                        # Each page chunk bboxes

                chunk_list.append({                                                                     # Add chunk to collection
                    "chunk_id": f"{stem}_chunk_{i}",
                    "chunk_type": chunk_type,
                    "text": enriched_text,
                    "parent_heading": heading,
                    "parent": parent
                })
                                 
            # Save ouput as markdown
            with open(f"{processed_path}/text/{stem}.md", "w", encoding="utf-8") as f:
                f.write(md_text)

            # Save chunks as json
            with open(f"{processed_path}/json/{stem}_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunk_list, f, indent=4, ensure_ascii=False)

            # Parse and save image for gpu colpali
            doc = pymupdf.open(pdf_path)
            for i, page in enumerate(doc):                                                          
                pixels = page.get_pixmap(dpi=300)                                                   
                temp_img_path = f"data/processed/temp_images/{stem}_page_{i+1}.png"
                pixels.save(temp_img_path)
                image_queue.put((temp_img_path, stem, i+1))                                           

        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            pdf_queue.task_done()