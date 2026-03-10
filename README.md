<div align="center">

# 🧬 MiniPat LLM — Evolutionary Patent Summarizer

**A RAG-based patent analysis pipeline powered by Mistral-7B and FAISS**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)](https://github.com/facebookresearch/faiss)

</div>

---

## 📌 Overview

**MiniPat LLM** is an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed to automatically analyze and summarize patent documents. It extracts text from patent PDFs using OCR, builds a semantic vector index, retrieves the most relevant passages, and generates structured patent summaries using a large language model — then evaluates the output quality against gold-standard references.

### ✨ Key Features

- 📄 **OCR-based PDF Extraction** — Handles scanned patent PDFs via Tesseract OCR
- 🔍 **Semantic Retrieval** — FAISS inner-product search over BGE-M3 embeddings
- 🤖 **LLM Generation** — Mistral-7B-Instruct for structured patent summarization
- 📊 **Multi-Metric Evaluation** — Cosine Similarity, ROUGE (1/2/L), and BERTScore
- ⚙️ **Configurable Pipeline** — Tunable chunk size, overlap, context length, and top-K retrieval

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EvoPat Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Patent PDF                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │ OCR Extract  │───▶│  Clean Text  │───▶│   Chunking   │      │
│   │  (Tesseract) │    │  (Stopwords) │    │ (300w / 60o) │      │
│   └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                           ┌──────────────┐      │
│                                           │  BGE-M3      │      │
│                                           │  Embeddings  │      │
│                                           └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│   │  Mistral-7B  │◀───│   Truncate   │◀───│ FAISS Index  │      │
│   │  Generation  │    │   Context    │    │  (Top-K=5)   │      │
│   └──────┬───────┘    └──────────────┘    └──────────────┘      │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────────────────────────┐              │
│   │              Evaluation Suite                 │              │
│   │  Cosine Similarity │ ROUGE-1/2/L │ BERTScore │              │
│   └──────────────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
evopat/
├── main.py                        # CLI entry point
├── README.md
├── Medical_Patents/               # Patent PDFs and gold-standard text files
│   ├── 1.pdf
│   ├── 1.txt
│   ├── 10.pdf
│   ├── 10.txt
│   └── ...
├── src/
│   ├── __init__.py
│   ├── config.py                  # Model names, hyperparameters, paths
│   ├── preprocessing.py           # PDF text extraction (OCR) & cleaning
│   ├── embeddings.py              # BGE-M3 sentence embeddings
│   ├── vectorstore.py             # FAISS index creation, save/load
│   ├── retriever.py               # Semantic similarity retrieval
│   ├── compression.py             # Context truncation
│   ├── generator.py               # Mistral-7B text generation
│   ├── pipeline.py                # End-to-end RAG pipeline + evaluation
│   └── evaluation/
│       ├── __init__.py
│       ├── similarity.py          # Cosine, ROUGE, BERTScore metrics
│       ├── rouge_eval.py
│       └── bert_score_eval.py
├── vectorstore/                   # Persisted FAISS index & metadata
│   ├── faiss_index.bin
│   └── metadata.pkl
└── data/                          # Additional data resources
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **CUDA-capable GPU** (required for Mistral-7B inference and embeddings)
- **Tesseract OCR** installed and available at `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/THE-Adi-005/EvoPat.git
   cd EvoPat/evopat
   ```

2. **Install dependencies**
   ```bash
   pip install torch transformers sentence-transformers faiss-gpu
   pip install PyMuPDF pytesseract Pillow nltk
   pip install rouge-score bert-score
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Install Tesseract OCR**
   - Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to `C:\Program Files\Tesseract-OCR\`

---

## 💻 Usage

### Basic Usage

Run the pipeline on a patent PDF with a query:

```bash
python main.py --pdf "Medical_Patents/10.pdf" --query "Provide a technical summary, core innovation, implementation details, and impact of this patent."
```

### Arguments

| Argument  | Type   | Required | Description                                    |
|-----------|--------|----------|------------------------------------------------|
| `--pdf`   | `str`  | ✅       | Path to the patent PDF file                    |
| `--query` | `str`  | ✅       | Query describing what to analyze in the patent |

### Example Output

```
Extracting text...
Raw length: 45230
Cleaning text for embedding...
Chunking...
Total chunks: 42
Embedding...
Creating FAISS index...
Retrieving relevant chunks...
Retrieved chunks: 5
Context words: 2500
Generating response...

===== MODEL OUTPUT =====

Technical Summary
...

Core Innovation
...

Implementation Details
...

Impact & Applications
...

===== SIMILARITY SCORES =====
Cosine: 0.8234
Rouge: {'rouge1': 0.42, 'rouge2': 0.18, 'rougeL': 0.35}
BERT: 0.8712
```

---

## ⚙️ Configuration

All hyperparameters are centralized in [`src/config.py`](src/config.py):

| Parameter           | Default                             | Description                                     |
|---------------------|-------------------------------------|-------------------------------------------------|
| `LLM_MODEL`         | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID for text generation      |
| `EMBEDDING_MODEL`   | `BAAI/bge-m3`                       | Sentence embedding model for retrieval          |
| `FAISS_INDEX_PATH`  | `vectorstore/faiss_index.bin`       | Path to persist the FAISS index                 |
| `METADATA_PATH`     | `vectorstore/metadata.pkl`          | Path to persist chunk metadata                  |
| `MAX_CONTEXT_WORDS` | `2500`                              | Maximum words in the retrieved context          |
| `TOP_K`             | `5`                                 | Number of top chunks to retrieve                |
| `CHUNK_SIZE`        | `300`                               | Words per chunk for text splitting              |
| `CHUNK_OVERLAP`     | `60`                                | Overlapping words between consecutive chunks    |

---

## 🧩 Module Reference

### `preprocessing.py`
- **`extract_text(pdf_path)`** — Renders each PDF page at 300 DPI and runs Tesseract OCR to extract text.
- **`clean_text(text)`** — Removes URLs, special characters, and English stopwords for cleaner embeddings.

### `embeddings.py`
- **`embed_texts(texts)`** — Encodes text chunks into normalized 1024-dim vectors using BGE-M3 on GPU.

### `vectorstore.py`
- **`create_index(dimension)`** — Creates a FAISS inner-product index.
- **`save_index(index, metadata)`** / **`load_index()`** — Persist and reload the FAISS index + metadata.

### `retriever.py`
- **`retrieve(query, index, metadata)`** — Embeds the query, searches the FAISS index, and returns the top-K most relevant chunks.

### `compression.py`
- **`truncate_context(text)`** — Truncates retrieved context to `MAX_CONTEXT_WORDS` to fit within the LLM's token budget.

### `generator.py`
- **`generate_response(prompt, max_new_tokens=600)`** — Runs greedy decoding on Mistral-7B-Instruct with FP16 precision.

### `pipeline.py`
- **`chunk_text(text, chunk_size, overlap)`** — Splits text into overlapping word-level chunks.
- **`run_pipeline(pdf_path, query)`** — Orchestrates the full RAG pipeline: extract → clean → chunk → embed → index → retrieve → generate → evaluate.

### `evaluation/similarity.py`
- **`cosine_similarity(text1, text2)`** — Embedding-based cosine similarity via BGE-M3.
- **`rouge_similarity(text1, text2)`** — ROUGE-1, ROUGE-2, and ROUGE-L F-measure scores.
- **`bert_similarity(text1, text2)`** — BERTScore F1 for semantic similarity.

---

## 📊 Evaluation Metrics

The pipeline evaluates generated summaries against gold-standard outputs using three complementary metrics:

| Metric              | What It Measures                          | Range   |
|---------------------|-------------------------------------------|---------|
| **Cosine Similarity** | Semantic embedding overlap (BGE-M3)     | 0.0–1.0 |
| **ROUGE-1/2/L**      | N-gram overlap (unigram, bigram, longest common subsequence) | 0.0–1.0 |
| **BERTScore**         | Contextual token-level semantic similarity | 0.0–1.0 |

---

## 🔬 Prompt Variants

The project supports multiple prompt strategies for comparative analysis:

| Prompt        | Role                       | Output Sections                                                        |
|---------------|----------------------------|------------------------------------------------------------------------|
| **Prompt 1**  | Professional patent examiner | Innovation, Abstract, Methodology, Results                            |
| **Prompt 2** *(active)* | Technical research analyst  | Technical Summary, Core Innovation, Implementation Details, Impact & Applications |
| **Prompt 3**  | Non-technical audience      | Key Invention, Plain Language Summary, How It Works, Significance     |

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|-----------------|-------------------------------------|
| Language        | Python 3.10+                        |
| LLM             | Mistral-7B-Instruct-v0.2           |
| Embeddings      | BAAI/bge-m3 (1024-dim)             |
| Vector Store    | FAISS (Inner Product)               |
| OCR             | Tesseract via PyMuPDF + pytesseract |
| Evaluation      | rouge-score, bert-score             |
| Deep Learning   | PyTorch, HuggingFace Transformers   |

---

## 📝 License

This project is for academic and research purposes.

---

<div align="center">

**Built with ❤️ for NLP Research**

</div>
