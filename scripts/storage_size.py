from pathlib import Path

from src.config import load_config

cfg = load_config()

pdf_path = cfg.dirs.raw_data
embedding_path = cfg.dirs.embeddings
processed_path = cfg.dirs.processed_data

# gather metrics
# n files, number of pages in each file, average number of pages, total number of pages
n_files = sum(1 for p in Path(pdf_path).rglob('*') if p.is_file) 

# total embeddings size, what res they are embedding at, how many patches, average embedding size

# dpi of temp images, average size of temp image

# average lines of text, total text size, average text size

# number of nodes in graph, total graph size, average graph size / pages of documents.



"""
if file_path.is_file():
    size = file_path.stat().st_size  # Size in bytes
    print(f"File size: {size} bytes")
"""