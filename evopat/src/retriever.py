import faiss
import numpy as np
from src.embeddings import embed_texts
from src.config import TOP_K

def retrieve(query, index, metadata):
    query_vec = embed_texts([query])
    scores, indices = index.search(query_vec, TOP_K)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results