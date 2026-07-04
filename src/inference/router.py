from openai import OpenAI
from src.config import config

model = config.models.vlm_model

router_prompt = """You are a query router for a scientific literature RAG system.
                    Classify the query into exactly one of: 'text', 'visual', or 'hybrid'.

                    - text: question answerable from text chunks alone
                    - visual: question requires interpreting a figure, graph, or diagram  
                    - hybrid: question requires both text and visual content

                    Output only the single word classification."""

def route_query(query, llm_client):
    response = llm_client.chat.completions.create(
        model = config.models.vlm_model,
        messages = [
                    {"role": "system", "content": router_prompt},
                    {"role": "user", "content": query}
                    ],

        temperature=0.0,
        max_tokens=10,
    )

    result = response.choices[0].message.content.strip().lower()

    # default to text if unexpected result
    if result not in ("text", "visual", "hybrid"):
        return "text"
    
    return result

