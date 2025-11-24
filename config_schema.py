from typing import Literal, Optional
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Ingestion
    chunk_size: int
    chunk_overlap: int
    
    # Indexing
    embedding_model: str
    
    # Retrieval
    retriever_method: Literal["dense", "sparse", "hybrid"]
    top_k: int
    
    # Advanced: Context Augmentation (Feature 1)
    # 0 = No augmentation, 1 = Prev + Next, 2 = 2 Prev + 2 Next
    context_window: int 
    
    # None = No reranking. String = Model Name.
    reranker_model: Optional[str] 
    
    # Generation
    llm_model: str
    prompt_style: str

    def __repr__(self):
        return (f"Config(chunk={self.chunk_size}, "
                f"retr={self.retriever_method}, k={self.top_k}, "
                f"win={self.context_window}, "
                f"rerank={ 'None' if not self.reranker_model else self.reranker_model.split('/')[-1] })")