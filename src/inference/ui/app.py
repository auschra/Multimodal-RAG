import streamlit as st
import requests
import json
import uuid
from pathlib import Path
from datetime import datetime

MEMORY_PATH = Path("memory.jsonl")
LLM_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer EMPTY"}

def load_memory():
    if not MEMORY_PATH.exists():
        return []
    records = []
    with MEMORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def save_memory(entry):
    with MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def call_llm(query, system_prompt="You are a helpful chatbot."):
    payload = {
        "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    resp = requests.post(LLM_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assistant_text = data["choices"][0]["message"]["content"]
    return data, assistant_text

st.set_page_config(page_title="RAG Query + Memory", layout="wide")
st.title("RAG — Query with Persistent Memory")

history = load_memory()
if "history" not in st.session_state:
    st.session_state.history = history

with st.sidebar:
    st.header("Memory")
    st.write(f"Saved entries: {len(st.session_state.history)}")
    if st.button("Clear memory (file only)"):
        MEMORY_PATH.unlink(missing_ok=True)
        st.session_state.history = []
        st.experimental_rerun()

query = st.text_area("Query", value="", height=120)
if st.button("Send Query"):
    with st.spinner("Calling LLM..."):
        try:
            raw, assistant_text = call_llm(query)
        except Exception as e:
            st.error(f"LLM request failed: {e}")
        else:
            entry = {
                "id": f"doc_{uuid.uuid4().hex[:8]}",
                "query": query,
                "assistant_text": assistant_text,
                "raw_response": raw,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            save_memory(entry)
            st.session_state.history.append(entry)
            st.success("Saved to memory.")
            st.write("Assistant:")
            st.info(assistant_text)

st.subheader("Conversation history")
for e in reversed(st.session_state.history[-50:]):
    st.markdown(f"**{e['timestamp']}** — `{e['id']}`")
    st.write("Q: " + e["query"])
    st.write("A: " + e["assistant_text"])
    st.divider()