# LLM-Based Root Cause Analysis for Virtualized 5G Network Logs

## Overview
This project presents an AI-driven Root Cause Analysis (RCA) system for virtualized 5G Core Networks. It combines log preprocessing, semantic retrieval and Large Language Models (LLMs) to generate structured and interpretable diagnostic outputs.

## Key Features
- Log preprocessing and normalization
- Semantic embeddings using SentenceTransformer
- FAISS-based similarity search
- Retrieval-Augmented Generation (RAG)
- Chain-of-Thought (CoT) reasoning
- Interactive Streamlit interface

## System Architecture
1. Log ingestion from 5G components (AMF, SMF, UPF)
2. Preprocessing and feature extraction
3. Embedding generation and FAISS indexing
4. RAG-based retrieval
5. LLM-based root cause analysis
6. Output visualization via Streamlit UI

## How to Run
```bash
pip install -r requirements.txt
streamlit run src/app_streamlit.py
