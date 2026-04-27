import json
from pathlib import Path
import fitz  # pip install PyMuPDF
from src.config import load_config

cfg = load_config()

# Set up paths
# Make sure this matches the filename pattern in your cpu_worker
json_path = Path(cfg.dirs.processed_data) / "json" / "geom_chunks.json"
pdf_path = Path(cfg.dirs.raw_data) / "geom.pdf"
output_dir = Path(cfg.dirs.processed_data) / "debug_images"

output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the JSON chunks
if not json_path.exists():
    print(f"Error: {json_path} not found. Run the ingestion pipeline first.")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 2. Group chunks by page using the new provenance array structure
chunks_by_page = {}
for chunk in chunks:
    provenance_list = chunk.get("provenance", [])
    for prov in provenance_list:
        page_num = prov.get("page_number")
        bbox = prov.get("bbox")
        
        if page_num and bbox:
            # Store the chunk metadata alongside its specific bbox for this page
            chunks_by_page.setdefault(page_num, []).append((chunk, bbox))

# 3. Open the PDF and Draw
doc = fitz.open(pdf_path)

# Process only the first 10 pages for debugging
for page_idx in range(min(10, len(doc))):
    page_num_1_indexed = page_idx + 1 
    page = doc[page_idx]
    
    page_chunks = chunks_by_page.get(page_num_1_indexed, [])
    print(f"Processing page {page_num_1_indexed}: Found {len(page_chunks)} bboxes.")
    
    for chunk, bbox in page_chunks:
        # bbox is [Left, Top, Right, Bottom] in normalized Top-Left space
        x0, y0, x1, y1 = bbox
        
        # PyMuPDF fitz.Rect(x0, y0, x1, y1) works directly with Top-Left
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # Draw the red box
        page.draw_rect(rect, color=(1, 0, 0), width=1.5)
        
        # Insert label (chunk_id or type)
        label = chunk.get("chunk_id", "chunk")
        # Position label slightly above the box
        page.insert_text(
            (rect.x0, max(5, rect.y0 - 2)), 
            label, 
            color=(0, 0, 1), 
            fontsize=6
        )

    # Save at 150 DPI for quick inspection
    pix = page.get_pixmap(dpi=150)
    output_file = output_dir / f"geom_page_{page_num_1_indexed}_bboxes.png"
    pix.save(str(output_file))

doc.close()
print(f"Debug images saved to: {output_dir}")