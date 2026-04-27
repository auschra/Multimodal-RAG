import requests

model = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"

system_prompt = """You are an agentic router. Your only job is to select the appropriate processing pipeline given the prompt. Your options are 'Colpali', 'Standard' or 'Hybrid'.
                Provide your reasoning for why you made that decision. Query:"""

def route_query(query):
    
    response = requests.post("http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        
        json={
            "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "user", "content": "Output your final classisfication. "}
                ],
            "max_tokens": 2048,
            "temperature": 0.0,             
        })

    out = response.json()
    text = out["choices"][0]["message"]["content"]
    print(text)
    
    return text


