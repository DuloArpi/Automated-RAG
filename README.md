# Automated RAG Pipeline Optimizer

This is a framework for Retrieval-Augmented Generation. Instead of guessing the best chunk size, retriever, or embedding model for your documents, AutoRAG automatically generates synthetic test data, evaluates thousands of pipeline combinations, and **exports a production-ready, Dockerized API** of the winning configuration.

## 🚀 Features

*   **Universal Ingestion**: Supports PDFs, Text, Markdown, and HTML.
*   **Automatic Optimization**: Uses Sequential or Grid Search to tune:
    *   **Chunk Sizes** (128, 256, 512, 1024)
    *   **Retrieval Strategies** (Dense vs. Sparse BM25 vs. Hybrid)
    *   **Rerankers** (MS-Marco, BGE-Reranker, None)
    *   **Context Augmentation** (Prev-Next Window Expansion)
    *   **Prompt Styles** (Default vs. Concise)
*   **Synthetic Evaluation**: Automatically generates a "Gold Standard" QA test set from your data using an LLM.
*   **Deployment Ready**: Exports the winning pipeline as a standalone **FastAPI** server with a **Dockerfile**.
*   **GPU Support**: Automatically detects CUDA (Nvidia) or MPS (Apple Silicon).

---

## 🛠️ Installation

```bash
git clone https://github.com/DuloArpi/automated-rag.git
cd autorag

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install langchain langchain-community langchain-openai langchain-huggingface chromadb rank_bm25 sentence-transformers transformers accelerate bitsandbytes pypdf unstructured tiktoken ragas datasets fastapi uvicorn
```

## Quick Start

### 1. Prepare Data
Create a folder named `my_data` in the root directory and drop your documents (PDFs, TXT) inside.

### 2. Run the Optimizer
Run the main script. This will generate test questions, run experiments, and find the best architecture for your specific data.

```bash
# Recommended: Fast Sequential Search (Phase-by-Phase optimization)
python main.py --mode sequential

# Optional: Exhaustive Grid Search (Slow, tries every combination)
python main.py --mode grid
```
### 3. Deploy the winner
Once the optimization finishes, the system creates a folder named best_rag_pipeline. This contains your fully optimized code.
```bash
cd best_rag_pipeline

# Install the dependencies for the winner
pip install -r requirements.txt

# Run the server
python server.py
```
The API is now live at http://localhost:8000.

### API Usage
You can query your optimized RAG pipeline via REST API:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is covered in these documents?"}'
```

### Configuration
You can adjust the search space or models in main.py:

```python
# Select models (Local or OpenAI)
GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
EVAL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# Adjust Search Space
CHUNK_SIZES = [256, 512, 1024]
RETRIEVERS = ["dense", "hybrid", "sparse"]
RERANKERS = [None, "BAAI/bge-reranker-base"]
CONTEXT_WINDOWS = [0, 1] # 1 = Add previous and next chunks to context
```
### Project Structure
```text
autorag/
├── autorag/               # Core framework code
│   ├── ingest/            # Loaders & Chunkers
│   ├── retrieval/         # Retrievers, Rerankers, Augmenters
│   ├── generation/        # LLM & Prompts
│   ├── optimizer/         # Genetic/Grid Search Engine
│   └── exporter/          # Docker/FastAPI Builder
├── main.py                # Entry point for optimization
└── best_rag_pipeline/     # (Generated) The final deployable bot
```


