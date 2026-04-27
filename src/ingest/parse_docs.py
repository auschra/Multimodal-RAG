import logging
from pathlib import Path
from src.clients.mineru_client import MinerUClient
import concurrent.futures

"""
json_format = {
  "chunk_id": "...",
  "document_id": "...",
  "text": "[contextual prefix]\n\n[chunk body]",
  "raw_text": "[chunk body only, no prefix]",
  "heading_path": ["Methods", "Sample Preparation"],
  "page_indices": [3, 4],                    # chunks can span pages
  "bboxes": [{"page": 3, "bbox": [...]}, ...], # one per source block
  "block_types": ["text", "text", "equation"],
  "source_block_ids": [17, 18, 19]
}"""


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pass pdf through mineru client, extract content_list.json as result
def parse_file(pdf_path, client):
    parse_output = client.parse(pdf_path)
    content_list = client.read_content_list(parse_output.output_dir, pdf_path.stem)         # writes to file
    
    return content_list

def main():
    client = MinerUClient()
    corpus_dir = Path("/storage/bulk/raw_docs")
    pdfs = sorted(f for f in corpus_dir.iterdir() if f.suffix.lower() == ".pdf")
    logger.info(f"{len(pdfs)} documents")
    
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

        jobs = {executor.submit(parse_file, pdf, client): pdf for pdf in pdfs}

        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            pdf = jobs[job]
            
            try:
                job.result()
                logger.info(f"[{i}/{len(pdfs)}] Parsed {pdf.name}")

            except Exception as e:
                failed.append((pdf, e))
                logger.error(f"[{i}/{len(pdfs)}] Failed {pdf.name}: {e}")
    
    logger.info(f"Done. {len(pdfs) - len(failed)} succeeded, {len(failed)} failed.")

if __name__ == "__main__":
    main()