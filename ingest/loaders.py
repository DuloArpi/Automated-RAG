import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Universal loader that selects the correct loading strategy based on file extension.
    """
    
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".html": UnstructuredHTMLLoader,
    }

    def load_file(self, file_path: str) -> List[Document]:
        """Loads a single file and returns a list of LangChain Documents."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.LOADER_MAPPING:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        loader_class = self.LOADER_MAPPING[ext]
        try:
            loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_directory(self, directory_path: str) -> List[Document]:
        """Recursively loads all supported files in a directory."""
        docs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1].lower() in self.LOADER_MAPPING:
                    docs.extend(self.load_file(file_path))
        return docs
    