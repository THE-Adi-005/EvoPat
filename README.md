# EvoPat – Patent Analysis System using RAG (Mistral)

EvoPat is a Retrieval-Augmented Generation (RAG) based patent intelligence system built using the Mistral large language model.
It enables contextual question-answering, structured summarization, and analytical insights over patent documents.

The system combines dense semantic retrieval with LLM-based reasoning to provide grounded and reliable outputs.

---

## 🚀 Overview

Traditional LLMs hallucinate when asked domain-specific questions.
EvoPat solves this by:

1. Extracting text from patent PDFs
2. Splitting into semantic chunks
3. Generating embeddings
4. Storing them in a vector database
5. Retrieving top relevant chunks
6. Passing context to Mistral for grounded generation

This ensures factual consistency and domain alignment.

---

## 🧠 Architecture

Patent PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ Vector Store
→ Similarity Retrieval
→ Mistral LLM
→ Context-Aware Response

---

## 🛠 Tech Stack

* Python 3.11
* Mistral (LLM)
* LangChain
* FAISS (Vector Store)
* Sentence Transformers / Embedding Model
* FastAPI (if deployed as API)
* PyMuPDF / pdf2image + OCR (for PDF extraction)

---

## 📂 Project Structure

EvoPat/
│── evopat/              # Core RAG logic
│── test_mistral.py      # Testing pipeline
│── requirements.txt
│── .env
│── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/THE-Adi-005/EvoPat.git
cd EvoPat

### 2️⃣ Create Virtual Environment

python -m venv venv311
venv311\Scripts\activate   (Windows)

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Add Environment Variables

Create a `.env` file and add:

MISTRAL_API_KEY=your_api_key_here

---

## ▶️ Run the System

python test_mistral.py

---

## 💡 Features

* Context-grounded patent Q&A
* Section-wise summarization (Abstract, Claims, Methodology)
* Reduced hallucination using retrieval grounding
* Modular and extensible pipeline
* Supports scalable document indexing

---

## 📊 Why RAG?

Without RAG:
LLM answers from pretraining → risk of hallucination

With RAG:
LLM answers using retrieved patent context → factual grounding

This makes the system suitable for:

* Patent review
* Novelty analysis
* Technical summarization
* Prior art search assistance

---

## 🔮 Future Improvements

* Hybrid search (BM25 + Dense retrieval)
* Cross-document similarity scoring
* Plagiarism detection layer
* Evaluation metrics (Recall@K, MRR)
* Frontend dashboard

---

## 👨‍💻 Author

Adithya R
Student – Amrita Vishwa Vidyapeetham
Interested in NLP, Retrieval Systems, and Applied AI

---

## 📜 License

For academic and research purposes.
