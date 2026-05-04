# =============================================================================
# retriever.py
#
# Purpose:
#   This module provides a function to retrieve the top-k most semantically similar log messages
#   to a user query using a FAISS index and SentenceTransformer embeddings. It is used as part of
#   the RAG pipeline to efficiently find relevant logs for root cause analysis.
# =============================================================================

import faiss
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Cache the model and index to avoid reloading them on every request
_model = None
_index = None
_logs_df = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_index():
    global _index
    if _index is None:
        index_path = "outputs/faiss.index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run src/faiss_index.py first.")
        _index = faiss.read_index(index_path)
    return _index

def get_logs_df():
    global _logs_df
    if _logs_df is None:
        logs_path = "outputs/feature_store.csv"
        if not os.path.exists(logs_path):
            raise FileNotFoundError(f"Feature store not found at {logs_path}. Please run src/preprocess.py first.")
        _logs_df = pd.read_csv(logs_path)
    return _logs_df

def get_top_k_logs(query: str, k: int = 3):
    # Encodes the user query into a dense vector using SentenceTransformer.
    model = get_model()
    query_vec = model.encode([query])

    # Loads the FAISS index from disk and searches for the top-k closest embeddings.
    index = get_index()
    D, I = index.search(query_vec, k)

    # Loads the log messages from the feature store to get all context columns
    logs_df = get_logs_df()
    
    # Returns a list of tuples: (distance, full_log_row_dict) for the top-k results.
    results = []
    for d, i in zip(D[0], I[0]):
        row = logs_df.iloc[i].to_dict()
        results.append((float(d), row))
    
    return results