import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

def download_arxiv_corpus():
    output_dir = Path("/storage/bulk/raw_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. The 8 Structural Stress-Test Papers
    known_ids = [
        "2006.11239",  # DDPM (Heavy Math)
        "2307.09288",  # Llama 2 (Massive Tables)
        "1512.03385",  # ResNet (Two-column standard)
        "2303.18223",  # LLM Survey (Massive bibliography filter)
        "2106.09685",  # LoRA (Algorithms / Pseudocode)
        "cs/9611048",  # LeCun 1998 (Old scan / OCR fallback)
        "2002.08264",  # Molecule Attention (Biochemical figures)
        "2205.14135",  # FlashAttention (Extremely dense footnotes)
    ]

    # ArXiv strictly requires a User-Agent, otherwise it will return a 403 Forbidden
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # 2. Fetch the 2 most recent cs.LG papers via API
    print("Querying ArXiv API for 2 recent cs.LG papers...")
    api_url = "http://export.arxiv.org/api/query?search_query=cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=2"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        recent_ids = []
        
        for entry in root.findall('atom:entry', ns):
            id_url = entry.find('atom:id', ns).text
            # Extract just the ID (e.g., http://arxiv.org/abs/2604.12345v1 -> 2604.12345v1)
            arxiv_id = id_url.split('/abs/')[-1]
            recent_ids.append(arxiv_id)
            print(f"Found recent paper: {arxiv_id}")
            
    except Exception as e:
        print(f"Failed to fetch recent papers from API: {e}")
        recent_ids = []

    all_ids = known_ids + recent_ids

    # 3. Download the PDFs
    print(f"\nStarting download of {len(all_ids)} papers...")
    for arxiv_id in all_ids:
        # Sanitize the ID for the filesystem (cs/9611048 -> cs_9611048.pdf)
        safe_name = arxiv_id.replace('/', '_')
        file_path = output_dir / f"{safe_name}.pdf"

        if file_path.exists():
            print(f"Skipping {safe_name}.pdf (Already exists)")
            continue

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"Downloading {arxiv_id}...")
        
        try:
            pdf_response = requests.get(pdf_url, headers=headers, stream=True, timeout=15)
            pdf_response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # ArXiv will IP-ban you if you do not rate limit automated downloads
            time.sleep(3)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {arxiv_id}: {e}")

if __name__ == "__main__":
    download_arxiv_corpus()