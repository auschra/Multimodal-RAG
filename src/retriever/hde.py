import requests
#from FlagEmbedding import BGEM3FlagModel       # broken with transformers > 5

"""
Hypothetical document emebeddings used to generate documents that could plausibly answer the query. Embedding these generated documents provide better
comparison against real documents for cosine sim that embedding query directly. 
"""

test_query = "What is the age of Domenic Rafferton?"

def gen_hde_doc(query):

    system_prompt = """You are a Hypothetical Document Embedder agent. Your job is to generate the document for the query. 
    Write a 1-2 sentence factual passage that directly answers the following query. 
    Write the passage exactly as it would appear in a reliable reference document. Do not include conversational filler or introductory text."""
    #model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True).                      reuse when fixed

    try:
        response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},

        json={
            "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
            "messages": [
                {"role": "system", "content": (system_prompt)},
                {"role": "user", "content": query}
            ],
            "max_tokens": 200,
            "temperature": 0.5
        }
        )
        response.raise_for_status()
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return ""
    
    except (KeyError, IndexError) as e:
        print(f"Failed to parse response: {e}")
        return ""

if __name__ == '__main__':
    out = gen_hde_doc(test_query)
    print(out)