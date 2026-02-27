import numpy as np
from src.evaluation.similarity import cosine_similarity, rouge_similarity, bert_similarity
from src.preprocessing import extract_text, clean_text
from src.embeddings import embed_texts
from src.vectorstore import create_index
from src.retriever import retrieve
from src.generator import generate_response
from src.compression import truncate_context
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

    print("Cleaning text for embedding...")
    cleaned_for_embed = clean_text(raw_text)

    print("Chunking...")
    chunks_embed = chunk_text(cleaned_for_embed)
    chunks_raw = chunk_text(raw_text)
    print("Total chunks:", len(chunks_embed))

    print("Embedding...")
    embeddings = embed_texts(chunks_embed)

    print("Creating FAISS index...")
    index = create_index(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    print("Retrieving relevant chunks...")
    retrieved = retrieve(query, index, chunks_raw)
    print("Retrieved chunks:", len(retrieved))

    context = " ".join(retrieved)
    context = truncate_context(context)

    print("Context words:", len(context.split()))

    # ---- PROMPT ----
    prompt = f"""
You are a professional patent examiner.

Use ONLY the provided PATENT CONTEXT.
Do NOT copy large verbatim text.
Do NOT fabricate unrelated inventions.

PATENT CONTEXT:
{context}

Generate the following sections clearly and completely:

Introduction:
(4 to 5 sentences)

Abstract:
(3 to 4 sentences)

Methodology:
(6 to 8 sentences)

Results:
(3 to 5 sentences)

After the above sections, provide evaluation strictly in valid JSON format.

Each JSON text field must contain exactly 2 concise sentences.
Scores must be integers between 0 and 10.

Return ONLY the sections followed by JSON.

{{
    "novelty_score": 0,
    "inventive_step_score": 0,
    "overlap_analysis": "",
    "key_differences": "",
    "technical_insights": "",
    "future_research_direction": ""
}}
"""

    print("Generating response...")
    response = generate_response(prompt)

    print("\n===== MODEL OUTPUT =====\n")
    print(response)

    # ---- EVALUATION ----
    gold_output = """
Separating Ions at and Above Atmospheric Pressure.
Innovation: Advanced ion separation mechanism at atmospheric pressure improving mass spectrometry efficiency.
Abstract: Systems and methods for ion separation using electric field manipulation at atmospheric pressure.
Methodology: Ion generation, electric field application, pressure-compatible ion optics, and integration with mass spectrometers.
Results: Improved ion transmission efficiency and analytical precision.
"""

    cos_sim = cosine_similarity(response, gold_output)
    rouge_sim = rouge_similarity(response, gold_output)
    bert_sim = bert_similarity(response, gold_output)

    print("\n===== SIMILARITY SCORES =====")
    print(f"Cosine: {cos_sim}")
    print(f"Rouge: {rouge_sim}")
    print(f"BERT: {bert_sim}")

    return response