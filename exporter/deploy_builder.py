import os
import shutil
import json
import dataclasses
import pathlib
from config_schema import RAGConfig

class DeployBuilder:
    """
    Exports a RAGConfig into a runnable FastAPI application with Docker support.
    """

    def export(self, config: RAGConfig, output_dir: str = "best_rag_pipeline"):
        
        # 1. Prepare Output Directory
        output_path = pathlib.Path(output_dir).resolve()
        
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. Copy the 'autorag' library Source Code
        # Logic: Find where this file is, go up two levels to find the 'autorag' package root
        current_file = pathlib.Path(__file__).resolve()
        package_root = current_file.parent.parent 

        dest_autorag = output_path / "autorag"

        print(f"    > Copying source code from: {package_root}")
        print(f"    > To: {dest_autorag}")

        shutil.copytree(
            package_root, 
            dest_autorag,
            ignore=shutil.ignore_patterns(
                "__pycache__", 
                "*.pyc", 
                ".git", 
                ".idea", 
                ".vscode",
                "my_data",             
                output_path.name,      
                "main.py",             
                "test_*.py"            
            )
        )

        # 3. Save Config as JSON
        config_dict = dataclasses.asdict(config)
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

        # 4. Generate requirements.txt
        self._write_requirements(output_path)

        # 5. Generate Dockerfile
        self._write_dockerfile(output_path)

        # 6. Generate server.py
        self._write_server_py(output_path, config)
        
        print(f"Export complete! Runnable pipeline saved to: {output_path}")

    def _write_requirements(self, output_path: pathlib.Path):
        reqs = """
fastapi
uvicorn
langchain
langchain-community
langchain-openai
langchain-huggingface
chromadb
rank_bm25
sentence-transformers
transformers
accelerate
bitsandbytes
pypdf
unstructured
tiktoken
ragas
datasets
        """
        with open(output_path / "requirements.txt", "w") as f:
            f.write(reqs.strip())

    def _write_dockerfile(self, output_path: pathlib.Path):
        dockerfile = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        with open(output_path / "Dockerfile", "w") as f:
            f.write(dockerfile.strip())

    def _write_server_py(self, output_path: pathlib.Path, config: RAGConfig):
        # We handle the Optional reranker_model field correctly for string injection
        reranker_val = f'"{config.reranker_model}"' if config.reranker_model else "None"

        server_code = f"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import AutoRAG components
from autorag.config_schema import RAGConfig
from autorag.ingest.loaders import DocumentLoader
from autorag.ingest.chunkers import Chunker
from autorag.indexing.embeddings import EmbeddingProvider
from autorag.indexing.vector_store import VectorStoreManager
from autorag.retrieval.retrievers import RetrieverFactory
from autorag.retrieval.rerankers import Reranker
from autorag.retrieval.augmentation import ContextAugmenter
from autorag.generation.llm import LLMFactory
from autorag.generation.prompts import PromptFactory
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="AutoRAG Deployed Pipeline")

# --- 1. Global State ---
vector_store = None
retriever = None
reranker = None
augmenter = None
llm = None
prompt_template = None
config = None

# --- 2. Startup Logic ---
@app.on_event("startup")
def startup_event():
    global vector_store, retriever, reranker, augmenter, llm, prompt_template, config
    
    print("Initializing Pipeline...")
    
    # Define Config
    config = RAGConfig(
        chunk_size={config.chunk_size},
        chunk_overlap={config.chunk_overlap},
        embedding_model="{config.embedding_model}",
        retriever_method="{config.retriever_method}",
        top_k={config.top_k},
        context_window={config.context_window},
        reranker_model={reranker_val},
        llm_model="{config.llm_model}",
        prompt_style="{config.prompt_style}"
    )
    
    # A. Ingest Documents
    doc_dir = "documents"
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        with open(os.path.join(doc_dir, "readme.txt"), "w") as f:
            f.write("Place your PDFs or Text files here.")
            
    loader = DocumentLoader()
    raw_docs = loader.load_directory(doc_dir)
    print(f"Loaded {{len(raw_docs)}} documents.")
    
    if not raw_docs:
        print("Warning: No documents found. Pipeline will be empty.")
        from langchain_core.documents import Document
        chunks = [Document(page_content="No data loaded.", metadata={{}})]
    else:
        # B. Chunk
        chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = chunker.split(raw_docs)
    
    # C. Embed & Index
    embed_provider = EmbeddingProvider()
    embed_model = embed_provider.get_embedding_model(config.embedding_model)
    
    vs_manager = VectorStoreManager(embed_model)
    vector_store = vs_manager.create_vector_store(chunks, collection_name="deployed_idx")
    
    # D. Retriever
    # If using a reranker, we fetch more docs initially (2x) to let the reranker filter
    fetch_k = config.top_k * 2 if config.reranker_model else config.top_k
    
    retriever = RetrieverFactory.create_retriever(
        vector_store=vector_store,
        documents=chunks,
        method=config.retriever_method,
        k=fetch_k
    )
    
    # E. Augmenter (Prev-Next)
    augmenter = ContextAugmenter(all_chunks=chunks)
    
    # F. Reranker
    if config.reranker_model:
        reranker = Reranker(model_name=config.reranker_model)
        
    # G. LLM
    llm = LLMFactory.create_llm(config.llm_model)
    prompt_template = PromptFactory.get_prompt(config.prompt_style)
    
    print("Pipeline Ready!")

# --- 3. API Endpoints ---

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    q = request.question
    
    # 1. Retrieve
    docs = retriever.invoke(q)
    
    # 2. Rerank
    if reranker:
        docs = reranker.rerank(q, docs, top_n=config.top_k)
    else:
        docs = docs[:config.top_k]
        
    # 3. Augment (Context Window)
    if config.context_window > 0:
        docs = augmenter.augment(docs, window_size=config.context_window)
        
    # 4. Generate
    context_str = "\\n\\n".join([d.page_content for d in docs])
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({{"context": context_str, "question": q}})
    except Exception as e:
        answer = f"Error generating answer: {{e}}"
    
    return {{
        "question": q,
        "answer": answer,
        "sources": [d.page_content for d in docs]
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open(output_path / "server.py", "w") as f:
            f.write(server_code.strip())