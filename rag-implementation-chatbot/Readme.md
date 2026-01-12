# Project RAG AI (Project_RAG_AI)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored for lecture/audio content. It includes tools to convert video/audio into text chunks, create embeddings, store them as JSON, and query a language model (LLM) with retrieved context.

## Pipeline (high level)
1. Ingest media: `Video_to_Mp3.py` / `Mp3_to_Jsons.py` extract audio and segment into manageable clips.
2. Transcription: convert audio segments to text using a speech-to-text tool (Whisper, Google STT, or other).
3. Chunking: split long transcripts into chunks with overlap for context preservation.
4. Embedding: generate vector embeddings per chunk (sentence-transformers or OpenAI embeddings).
5. Store: save chunk texts and embeddings to JSON files (seen in `jsons/` and `newjsons/`).
6. Retrieval: on a user query, compute query embedding, find nearest chunks (cosine similarity) and assemble context.
7. Generation: send retrieved context + query to an LLM for an answer (`using_gemini_api.py` demonstrates LLM usage).

## Files of note
- `Read_Jsons_and_create_embeddings.py`: reads text JSONs and computes embeddings.
- `process_incoming.py`: pipeline logic to accept new inputs and process them.
- `using_gemini_api.py`: example integration with an LLM API to generate responses using retrieved context.
- `audios/`, `jsons/`, `newjsons/`: storage for raw audio and processed JSONs.

## Design choices & rationale
- Chunking with overlap preserves local context across chunk boundaries.
- Embeddings allow semantic search instead of keyword matching, improving retrieval relevance.
- Keeping JSON-based storage is simple for small-scale projects; for production, use a vector DB (FAISS, Milvus, Pinecone).

## Dependencies & tools
- Python 3.8+
- ffmpeg (for audio extraction), pydub
- Speech-to-text: Whisper or cloud STT
- `sentence-transformers` or `openai` for embeddings
- numpy, pandas, scikit-learn (for similarity), streamlit (optional)

Install with:

```bash
pip install -r requirements.txt
```

## Run / Usage
- Follow these steps: convert media → transcribe → chunk → embed → query. Example scripts are provided in the repository.

## Notes & improvements
- Consider switching to a dedicated vector DB for large-scale retrieval.
- Add automated re-processing for new uploads and a permissions/metadata layer for datasets.
- For sensitive content, add filtering and redaction before storing transcripts.
# 🎓 RAG AI Teaching Assistant  
A Retrieval-Augmented Generation (RAG) system built on your own dataset of lecture videos using Whisper, embeddings, cosine similarity, and an LLM (OpenAI / DeepSeek local).

---

## 📂 Project Structure

PROJECT_RAG_AI/
│
├── audios/ # MP3 files extracted from videos
├── videos/ # Raw MP4 YouTube videos
├── jsons/ # Whisper transcription JSONs
├── newjsons/ # Cleaned/merged JSONs
├── unused/ # Optional discarded files
│
├── audio_cut.py # Script to trim long MP3 files
├── Video_to_Mp3.py # Convert mp4 → mp3
├── Mp3_to_Jsons.py # Run Whisper → JSON transcripts
├── merge_chunks.py # Merge chunked Whisper outputs
├── Read_Jsons_and_create_embeddings.py # Create embeddings + joblib
├── process_incoming.py # RAG pipeline (query → similarity → LLM)
│
├── embeddings.joblib # Vector store (embeddings)
├── config.py # API keys, model paths
├── prompt.txt # Base prompt template
├── response.txt # Output logging
├── output.json # Processed JSON output
└── Readme.md



---

## 🚀 Overview

This project allows you to turn long lecture videos into a searchable knowledge base, capable of answering questions using RAG.

Pipeline:

1. **Download videos** (MP4) from YouTube and place them inside `videos/`.
2. **Convert MP4 → MP3** using `Video_to_Mp3.py` (FFmpeg/Whisper backend).
3. **Trim long audios** to reduce transcription time using `audio_cut.py`  
   → e.g., keep only the first 10 minutes from 1–1.5 hour files.
4. **Transcribe MP3 → JSON** using Whisper via `Mp3_to_Jsons.py`.
5. **Process & clean JSONs** using `merge_chunks.py` or similar scripts.
6. **Create embeddings** for each transcript chunk using  
   `Read_Jsons_and_create_embeddings.py`, producing:
7. **At query time**, `process_incoming.py`:
- Converts user query into an embedding  
- Computes cosine similarity with stored embeddings  
- Retrieves top-k matching transcript chunks  
- Builds LLM prompt  
- Sends to OpenAI GPT or local DeepSeek model  
- Returns an answer grounded in your dataset

---

## 🧰 Technologies Used

| Component | Tool/Model |
|----------|------------|
| Video → Audio | FFmpeg |
| Speech-to-Text | Whisper (base/medium/large) |
| Embeddings | OpenAI / local embedding model |
| Similarity Search | Cosine similarity (NumPy / sklearn) |
| LLM | OpenAI GPT / DeepSeek locally downloaded |
| Storage | Pandas + Joblib |

---

## 📦 Installation
openai
whisper
ffmpeg-python
pandas
numpy
scikit-learn
joblib
tqdm
python
