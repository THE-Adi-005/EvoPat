import numpy as np
from src.preprocessing import extract_text, clean_text
from src.embeddings import embed_texts
from src.vectorstore import create_index
from src.retriever import retrieve
from src.generator import generate_response
from src.compression import truncate_context
from src.config import TOP_K

def chunk_text(text, chunk_size=350, overlap=75):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):       
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


def run_pipeline(pdf_path, query):
    print("Extracting text...")
    raw_text = extract_text(pdf_path)
    print("Raw length:", len(raw_text))

    print("Cleaning text...")
    cleaned = clean_text(raw_text)
    print("Cleaned length:", len(cleaned))

    print("Chunking...")
    chunks = chunk_text(cleaned)
    print("Total chunks:", len(chunks))

    print("Embedding...")
    embeddings = embed_texts(chunks)

    print("Creating FAISS index...")
    index = create_index(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    metadata = chunks

    print("Retrieving relevant chunks...")
    retrieved = retrieve(query, index, metadata)
    print("Retrieved chunks:", len(retrieved))

    context = " ".join(retrieved)
    context = truncate_context(context)

    print("Context words:", len(context.split()))

    prompt = f"""
You are a professional patent examiner.

Claim:
{query}

Respond strictly in valid JSON format.
Ensure ALL fields are filled.
Do NOT truncate the response.
End the response with a complete closing brace.

{{
  "novelty_score": 0-10,
  "inventive_step_score": 0-10,
  "overlap_analysis": "...",
  "key_differences": "...",
  "technical_insights": "...",
  "future_research_direction": "..."
}}
"""

    print("Generating response...")
    response = generate_response(prompt)

    return response


# Relevant Prior Art:
# {context}
