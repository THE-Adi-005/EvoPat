from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

def embed_texts(texts):
    return model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )   