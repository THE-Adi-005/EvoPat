import faiss
import pickle
import numpy as np
from src.config import FAISS_INDEX_PATH, METADATA_PATH

def create_index(dimension=1024):
    return faiss.IndexFlatIP(dimension)

def save_index(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata