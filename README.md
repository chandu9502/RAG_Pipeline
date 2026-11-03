#Retrieval Augmented Generation (RAG) Pipeline

A modular Retrieval-Augmented Generation (RAG) system built with **FAISS**, **Sentence-Transformers**, and **Groq LLM** (via LangChain). 

This pipeline ingests documents across multiple formats, creates normalized dense embeddings, stores them in a local FAISS index, and provides context grounded answers with cosine similarity thresholding

---

## Architecture Overview

```
[ Documents in data/ ] 
          │
          ▼
1. Document Ingestion (data_loader.py) ──> PDF, TXT, CSV, XLSX, DOCX, JSON
          │
          ▼
2. Chunking & Embedding (embedding.py) ──> RecursiveCharacterTextSplitter + all-MiniLM-L6-v2
          │
          ▼
3. Vector Indexing (vectorstore.py)   ──> FAISS (L2 + Cosine Similarity Scoring & Thresholding)
          │
          ▼
4. Retrieval & LLM (search.py)        ──> Top-K filtered chunks + Groq Chat LLM (e.g. gpt-oss-20b)
          │
          ▼
5. Entrypoints (app.py, src/)         ──> Interactive CLI
```

---

## Project Structure

```
RAG/
├── app.py                  # Main CLI application 
├── requirements.txt        # Python package dependencies
├── pyproject.toml          # Project configuration & metadata
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules for credentials, caches, & indices
├── data/                   # Input document repository
│   └── pdf_files/          # Source PDFs
└── src/                    # Core modular library
    ├── __init__.py         # Package exports & unified RAGPipeline class
    ├── data_loader.py      # Multi format document loaders
    ├── embedding.py        # Chunking & local SentenceTransformer embedding pipeline
    ├── vectorstore.py      # FAISS vector store
    ├── search.py           # RAG retrieval & Groq LLM synthesis with guardrails
    └── faiss_store/        # Persisted FAISS index 
```

---

## Key Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, and `.json` documents automatically.
- **Local Dense Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally (384-dimensional normalized vectors).
- **Exact Vector Search**: Fast similarity search using FAISS with dual persistence (`faiss.index` binary + `metadata.pkl`).
- **Relevance Filtering & Guardrails**:
  - Automatically converts FAISS Euclidean distances into cosine similarity:
    $$\text{similarity} = 1 - \frac{\text{dist}}{2}$$
  - Drops irrelevant results below `score_threshold` 
---


## Installation & Setup

1. **Clone or navigate to the repository:**
   ```powershell
   cd path/to/RAG
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

---

## Running the Application

### 1. Interactive CLI Mode
Run the assistant in an interactive question-and-answer loop:
```powershell
python app.py
```
```text
============================================================
RAG Pipeline Interactive Assistant
Type 'exit', 'quit', or 'q' to stop.
============================================================

Enter query: What is qubit?
Answer:
A qubit is the basic unit of quantum information...
```

### 2. Direct Query via Command-Line
Pass your query directly as an argument:
```powershell
python app.py "What is quantum entanglement?"
```






