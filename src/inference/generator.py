from openai import OpenAI
from src.config import config

system_prompt = """You are a scientific literature assistant with access to retrieved excerpts from academic papers. Your role is to answer questions by reasoning directly over the provided source material.

## Core Behaviour
- Ground every claim in the provided context chunks. Cite the chunk number inline when making a specific claim, e.g. [1], [2].
- Reason across sources — if multiple chunks address the same topic from different angles, synthesise them into a coherent answer rather than summarising each in isolation.
- Identify contradictions explicitly. If sources disagree, state this directly: "While [1] finds that X leads to Y under these conditions, [3] reports the opposite effect, potentially explained by differences in methodology."
- Draw conclusions where the evidence supports it. You are not a search engine — if the retrieved material collectively implies something, say so while attributing the reasoning to the sources.

## Handling Insufficient Context
If the retrieved context does not adequately cover the question, make this immediately clear at the start of your response:
"The retrieved source material does not directly address this question. The following answer draws on general scientific knowledge rather than the provided documents: ..."
Never silently answer from general knowledge without flagging it. The user must always know whether your answer is grounded in their documents or not.

## Formatting
- For straightforward factual questions, respond in concise prose.
- For complex questions requiring synthesis across multiple sources, use short paragraphs organised by sub-topic.
- Do not use bullet points unless listing genuinely enumerable items (e.g. experimental conditions, dataset statistics).
- Keep responses focused — do not pad with restating the question or summarising what you are about to say.

## Hard Constraints
- Do not fabricate citations or claim a chunk says something it does not.
- Do not present your own reasoning as if it were sourced from the documents.
- If a chunk is only tangentially relevant, note this rather than treating it as strong evidence."""

def format_chunks(chunks: list[dict])->str:
    # Format chunks returned by graph search so they can be added to context window.
    # Need to deduplicate chunk_ids, and sort by score desc, some context for the LLM of importance

    read = {}
    for chunk in chunks:

        # Check if chunk is new or if not, if new instance has higher score, add to list
        if chunk["chunk_id"] not in read or chunk.get("score", 0) > read[chunk["chunk_id"]].get("score", 0):
            read[chunk["chunk_id"]]

    deduped_chunks = sorted(read.values, key=lambda x: x.get("score", 0), reverse=True)
    
    # Format chunk content for context window
    lines = []
    for i, chunnk in enumerate(deduped_chunks, 1):
        # Find where the chunk came from (graph traversal or initial retrieval)
        source = "[graph_search]" if chunk.get("source") == "graph_hop" else "[retrieval]"

        # Get heading of chunking
        heading = f"| {chunk["heading"]}" if {chunk.get("heading")} else ""

        lines.append(
            f"[{i}] {source} doc: {chunk.get('document_id', 'unknown')}"
            f"{heading} (score: {chunk.get('score', 0):.2f})\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(lines)



def generate(query: str, chunks: list[dict], llm_client: OpenAI)-> dict:

    text_context = format(chunks)

    response = llm_client.chat.completions.create(
        model = config.models.vlm_model,
        messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{text_context}\n\nQuestion: {query}"},
        ],

        temperature = 0.4,
        max_tokens = config.token_limit,
    )

    result = response.choices[0].message.content.strip().lower()

    
    return {
        "answer": response.choices[0].message.content.strip(),
        "contexts": [c.get("text", "") for c in chunks],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    }


