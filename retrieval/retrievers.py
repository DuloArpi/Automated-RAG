from typing import List
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

# --- Custom Ensemble Implementation to bypass import errors ---
class SimpleEnsembleRetriever(BaseRetriever):
    """
    A simple implementation of an Ensemble Retriever that queries multiple retrievers
    and combines the results, deduplicating them.
    """
    retrievers: List[BaseRetriever]
    weights: List[float] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # Gather results from all retrievers
        all_docs = []
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error in sub-retriever: {e}")

        # Deduplicate based on page_content
        unique_docs = []
        seen_content = set()
        
        # Preserve the order
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
                
        return unique_docs

# --- Factory ---
class RetrieverFactory:
    """
    Creates a retriever based on the configuration.
    """

    @staticmethod
    def create_retriever(
        vector_store: VectorStore, 
        documents: List[Document], 
        method: str = "dense", 
        k: int = 5,
        weight_dense: float = 0.5,
        weight_sparse: float = 0.5
    ):
        """
        Args:
            vector_store: The populated vector store.
            documents: The original chunks (needed for BM25 index building).
            method: 'dense', 'sparse', or 'hybrid'.
            k: Number of documents to retrieve.
        """
        
        # 1. Dense Retriever (Vector Search)
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": k})

        if method == "dense":
            return dense_retriever

        # 2. Sparse Retriever (BM25)
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = k

        if method == "sparse":
            return sparse_retriever

        # 3. Hybrid Retriever (Ensemble)
        if method == "hybrid":
            return SimpleEnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[weight_dense, weight_sparse]
            )
        
        raise ValueError(f"Unknown retrieval method: {method}")