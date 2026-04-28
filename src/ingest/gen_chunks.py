from pathlib import Path
from typing import Iterator
import json
import uuid
from dataclasses import dataclass, field
import requests 
from openai import OpenAI
from transformers import AutoTokenizer
from itertools import zip_longest

# TOADD
# - Layout chunking 
#   - find and split based on tags
# - Contextual retrieval
    # single sentence appended to beginnning of chunk summarising the chunk and describing the chunks fit in the overall. contenxt
    # ie. this paragraph explains the returns of the company Nvidia in the previous year 2025 vs the current year 2026

class HybridChunker:
    def __init__(self):
        # Useless for chunking (page number?)
        self.JUNK_TYPES = {
        "page_number", "page_header", "page_footer", 
        "page_footnote", "header", "page_aside_text",
        "aside_text", "index",}

        self.llm_client = OpenAI(
            api_key = "EMPTY",
            base_url = "http://localhost:8001/v1"
        )
        self.model_name = "Qwen/Qwen3-32B-AWQ"
        
        self.context_prompt = """
            <document>
            [The ENTIRE text of your parsed PDF goes here]
            </document>

            Here is the chunk we want to situate within the whole document:
            <chunk>
            [The specific text of Chunk #1]
            </chunk>

            Please give a short, succinct context (1-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval.
            """
        self.token_limit = 1024

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', trust_remote_code=True)


    def get_token_count(self, text):
        n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        
        return n_tokens
    
    # Manage text based on token limit or layout tag
    def _iter_blocks(self, items):
        if isinstance(items, dict):
            yield items

        elif isinstance(items, list):
            for x in items:
                yield from self._iter_blocks(x)

    def gen_chunks(self, v2_content, doc_stem):
        """
        Hybrid chunker
        - new chunk on heading change or token limit
        """

        chunk_list = []                     # Confirmed chunks
        chunk_buffer = []                   # Stores blocks to create candidate chunk
        block_buffer = []                  # individual block metadatas (block_idx, type, page_idx, bbox)
        chunk_token_count = 0               # Counts tokens in candidate chunks (also blocks)
        current_level = None
        current_heading = None
        valid_text_blocks = {"paragraph", "text", "equation", "table", "list"}

        def flush_chunk():
            """
            Assign the current chunk buffer as a new chunk and reset.
            """
            nonlocal chunk_buffer, block_buffer, chunk_token_count                  # Redefine local vars
            if chunk_buffer:
                chunk_list.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": doc_stem,
                    "heading": current_heading,
                    "text": "\n".join(chunk_buffer),
                    "token_count": chunk_token_count,
                    "block_metas": block_buffer,
                })

            chunk_buffer = []
            block_buffer = []
            chunk_token_count = 0


        for page_idx, page in enumerate(v2_content):
            for block_idx, block in enumerate(page):
                block_type = block.get("type")
                bbox = block.get("bbox")
                block_meta = {
                    "block_idx": block_idx,
                    "page_idx": page_idx,
                    "block_type": block_type,
                    "bbox": bbox,
                }

            # 1. Headings (new chunk when hit new heading) 
            if block_type == 'title':                        # "type": "title", "content": {"title_content": [{"type": "text", "content": "**block_text**"], "level": 1,

                flush_chunk()
                current_heading = block.get('content').get('title_content')[0].get('content')
                
                continue        


            # 2. Text handling (+ token limtis)    ******** Handle other allowed block types
            if block_type == "paragraph":

                parts = block.get("content", {}).get("paragraph_content", [])
                text = " ".join(p.get("content", "") for p in parts if isinstance(p, dict) and p.get("type") == "text").strip()
                
                if not text:

                    continue

                block_token_count = self.get_token_count(text)

                # Check if new block would exceed token limit
                if chunk_buffer and chunk_token_count + block_token_count > self.token_limit:
                    flush_chunk()

                chunk_buffer.append(text)
                block_buffer.append(block_meta)
                chunk_token_count += block_token_count

                continue

            # SKIP OTHER TYPES FOR NOW

        # Flush last buffer of document
        flush_chunk()

        print(f"{len(chunk_list)} chunks created for {doc_stem}")

        return chunk_list

    # LLM call to append context summary to beginng of chunk
    def contextual_prefix(self, text, parent_doc):
        prompt = f"""
                <document> 
                {parent_doc} 
                </document> 

                Here is the chunk we want to situate within the whole document 
                <chunk> 
                {text} 
                </chunk> 

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
                """.strip()

        resp = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You write concise retrieval context. You must output the final answer immediately. DO NOT use <think> tags or output any internal reasoning."
                },
                {"role": "user", "content": prompt},
            ], 
            temperature=1.0,
            top_p=0.95,
            presence_penalty=0.0,
            extra_body={"top_k": 20,},
            max_tokens=120,
        )
        print(resp)
        
        return resp.choices[0].message.content.strip()

    def save_chunk(self, chunk, doc_stem):

        save_dir = Path("/datasets/scratch/02_chunks") 
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{doc_stem}.json"

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        return save_path
    

def main():
    chunker = HybridChunker()
    host_container = Path("/datasets/scratch/mineru-output/")                   # where to read mineru output from

    for doc_dir in host_container.iterdir():
        if not doc_dir.is_dir():
            continue
        
        json_path = next(doc_dir.glob("auto/*_content_list_v2.json"))
        idx_json_path = next(doc_dir.glob("auto/*_content_list.json"))              # Json containing pg_idx and bboxes  
        md_path = next(doc_dir.glob("auto/*.md"), None) 
        doc_stem = md_path.stem if md_path else json_path.stem.replace("_content_list_v2", "")
        parent_doc = md_path.read_text(encoding="utf-8") if md_path else ""                     # Full document markdown text

        with idx_json_path.open("r", encoding="utf-8") as f:
            idx_json = json.load(f)

        with json_path.open("r", encoding="utf-8") as f:
            content_json = json.load(f)
            chunk = chunker.gen_chunks(content_json, idx_json, doc_stem)
            save_path = chunker.save_chunk(chunk, doc_stem)

if __name__ == "__main__":
    main()