import shutil
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

class VectorStoreManager:
    """
    Manages the creation and retrieval of vector stores.
    For AutoRAG, we often create temporary stores for each experiment.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def create_vector_store(self, documents: List[Document], collection_name: str = None, persist_directory: str = None):
        """
        Ingests documents and returns a Chroma vector store.
        
        Args:
            documents: List of chunked documents.
            collection_name: Unique name for this experiment's index.
            persist_directory: If provided, saves to disk. If None, runs in-memory.
        """
        if not collection_name:
            collection_name = f"experiment_{uuid.uuid4().hex[:8]}"

        # If running in-memory (no persist_directory), Chroma is ephemeral.
        # If running on disk, we might want to clean up previous runs with same name logic if needed.
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        return vectorstore

    @staticmethod
    def clear_persisted_db(path: str):
        """Utility to clean up disk space after experiments."""
        try:
            shutil.rmtree(path)
            print(f"Deleted vector DB at {path}")
        except FileNotFoundError:
            pass