import glob
import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "newjsons"


def load_chunks() -> List[dict]:
    chunks = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data.get("chunks", []):
                chunk["source"] = path.name
                chunks.append(chunk)
    return chunks


def score_chunk(chunk_text: str, query: str) -> int:
    text_lower = chunk_text.lower()
    score = 0
    for token in query.lower().split():
        if token and token in text_lower:
            score += 1
    return score


def search(chunks: List[dict], query: str, top_k: int = 5) -> List[Tuple[dict, int]]:
    scored = []
    for chunk in chunks:
        s = score_chunk(chunk.get("text", ""), query)
        if s > 0:
            scored.append((chunk, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


st.set_page_config(page_title="Project RAG AI", page_icon="🤖", layout="wide")
st.title("Project RAG AI — Lecture QA")
st.markdown(
    "Use your lecture transcripts (newjsons) to retrieve relevant snippets. This app uses a simple keyword scorer so it runs without extra dependencies."
)

with st.sidebar:
    st.header("How to use")
    st.write("1. Type a question or keywords.\n2. Top matching lecture snippets will show.\n3. Refine your query for better matches.")
    st.write("Data folder:", DATA_DIR)

query = st.text_input("Ask a question or search keywords", placeholder="e.g., What are Python variables?")

if "chunks_cache" not in st.session_state:
    st.session_state["chunks_cache"] = load_chunks()

chunks = st.session_state["chunks_cache"]

if query:
    results = search(chunks, query)
    if not results:
        st.info("No matches found. Try different keywords.")
    else:
        for chunk, score in results:
            with st.expander(f"{chunk.get('Title', 'Lecture')} · score {score} · {chunk.get('start')}s to {chunk.get('end')}s"):
                st.write(chunk.get("text", "").strip())
                st.caption(f"Source: {chunk.get('source', '')}")
else:
    st.info("Enter a query to retrieve relevant lecture snippets.")
