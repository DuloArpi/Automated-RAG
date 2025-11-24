import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure parent folder is on sys.path so we can import config_schema from one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import AutoRAG components
from config_schema import RAGConfig
from ingest.loaders import DocumentLoader
from ingest.chunkers import Chunker
from indexing.embeddings import EmbeddingProvider
from indexing.vector_store import VectorStoreManager
from retrieval.retrievers import RetrieverFactory
from retrieval.rerankers import Reranker
from generation.llm import LLMFactory
from generation.prompts import PromptFactory
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="AutoRAG Deployed Pipeline")

# --- 1. Global State ---
vector_store = None
retriever = None
reranker = None
llm = None
prompt_template = None
config = None

# --- 2. Startup Logic ---
@app.on_event("startup")
def startup_event():
    global vector_store, retriever, reranker, llm, prompt_template, config
    
    print("Initializing Pipeline...")
    
    # Define Config (Hardcoded from Export)
    config = RAGConfig(
        chunk_size=128,
        chunk_overlap=20,
        embedding_model="all-MiniLM-L6-v2",
        retriever_method="hybrid",
        top_k=3,
        use_reranker=True,
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompt_style="concise"
    )
    
    # A. Ingest Documents
    # We look for a 'documents' folder in the deployed dir
    doc_dir = "documents"
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        print("Warning: 'documents' folder empty. Please add files and restart.")
        # Create a dummy file so it doesn't crash
        with open(os.path.join(doc_dir, "readme.txt"), "w") as f:
            f.write("Place your PDFs or Text files here.")
            
    loader = DocumentLoader()
    raw_docs = loader.load_directory(doc_dir)
    print(f"Loaded {len(raw_docs)} documents.")
    
    # B. Chunk
    chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.split(raw_docs)
    
    # C. Embed & Index
    embed_provider = EmbeddingProvider()
    embed_model = embed_provider.get_embedding_model(config.embedding_model)
    
    vs_manager = VectorStoreManager(embed_model)
    vector_store = vs_manager.create_vector_store(chunks, collection_name="deployed_idx")
    
    # D. Retriever
    retriever = RetrieverFactory.create_retriever(
        vector_store=vector_store,
        documents=chunks,
        method=config.retriever_method,
        k=config.top_k
    )
    
    # E. Reranker
    if config.use_reranker:
        reranker = Reranker()
        
    # F. LLM
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
        
    # 3. Generate
    context_str = "\n\n".join([d.page_content for d in docs])
    chain = prompt_template | llm | StrOutputParser()
    
    answer = chain.invoke({"context": context_str, "question": q})
    
    return {
        "question": q,
        "answer": answer,
        "sources": [d.page_content for d in docs]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)