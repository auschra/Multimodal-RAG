import multiprocessing as mp
from pathlib import Path
from src.ingest.cpu_worker import docling_worker
from src.ingest.gpu_worker import colpali_worker
from src.config import load_config

cfg = load_config()

pdf_path = cfg.dirs.raw_data
processed_path = cfg.dirs.processed_data
embeddings_path = cfg.dirs.embeddings


# Coordinate PDF ingestion pipelines (CPU & GPU) 
def ingest_pipeline(pdf_path):

    # PDF queue for cpu text ingestion and temporary image creation
    pdf_queue = mp.JoinableQueue()
    image_queue = mp.JoinableQueue()

    # Add PDFs to queue
    pdf_paths = list(Path(pdf_path).glob("*.pdf"))
    for path in pdf_paths:
        pdf_queue.put(str(path))

    # Create CPU workers
    num_cpu_workers = 20
    cpu_processes = []
    for _ in range(num_cpu_workers):
        p = mp.Process(target=docling_worker, args=(pdf_queue, image_queue))
        p.start()
        cpu_processes.append(p)

    # Create GPU worker (single)
    gpu_process = mp.Process(target=colpali_worker, args=(image_queue,))
    gpu_process.start()

    # Wait until CPU workers finished
    pdf_queue.join() 

    # PP -> stop CPU workers
    for _ in range(num_cpu_workers):
        pdf_queue.put(None)
    for p in cpu_processes:
        p.join()

    # Wait until GPU worker finished
    image_queue.join()
    
    # PP -> stop GPU worker
    image_queue.put(None)
    gpu_process.join()

    print("Ingestion phase complete.")

if __name__ == "main":
    ingest_pipeline(pdf_path)