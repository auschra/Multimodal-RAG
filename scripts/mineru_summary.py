import json
from pathlib import Path
from collections import Counter

v2 = json.loads(Path(
    "/datasets/scratch/mineru-output/2604.20824v1/2604.20824v1/auto/2604.20824v1_content_list_v2.json"
).read_text())

# Flatten, tagging each block with its page index
flat = []
for page_idx, page_blocks in enumerate(v2):
    for block in page_blocks:
        block = dict(block)
        block["_page_idx"] = page_idx
        flat.append(block)

print(f"Total blocks: {len(flat)}")
print(f"\nOuter types: {Counter(b['type'] for b in flat)}")

# For each distinct outer type, show one full example
seen = set()
print("\n=== One example per outer type ===")
for b in flat:
    if b["type"] not in seen:
        seen.add(b["type"])
        print(f"\n--- type='{b['type']}' ---")
        print(json.dumps(b, indent=2)[:600])