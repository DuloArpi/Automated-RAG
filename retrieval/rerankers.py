import torch
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Uses a Cross-Encoder model to re-score and re-order retrieved documents.
    Supports selectable models.
    """
    
    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("Reranker initialized with empty model name")

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"Loading Reranker on {device.upper()}: {model_name}...")
        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            print(f"Error loading reranker {model_name}: {e}")
            raise e

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        if not documents:
            return []

        # Prepare input pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict
        scores = self.model.predict(pairs)
        
        # Sort
        scored_docs = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top_n
        top_docs = [doc for doc, score in scored_docs[:top_n]]
        return top_docs