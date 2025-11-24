from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document

class Chunker:
    """
    Handles the splitting of documents and adds structural metadata
    (source, chunk_index) to enable Prev-Next augmentation.
    """
    
    def __init__(self, method: str = "recursive", chunk_size: int = 512, chunk_overlap: int = 50):
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        if self.method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # 1. Group documents by source file so we can index them sequentially
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)

        # 2. Split and Index
        final_chunks = []
        
        for source, docs in docs_by_source.items():
            # Split this file's content
            file_chunks = splitter.split_documents(docs)
            
            # Tag them with sequential IDs
            for i, chunk in enumerate(file_chunks):
                chunk.metadata["source"] = source
                chunk.metadata["chunk_index"] = i
                # Store total chunks in this file to prevent out-of-bounds errors later
                chunk.metadata["total_chunks_in_file"] = len(file_chunks)
                final_chunks.append(chunk)

        return final_chunks