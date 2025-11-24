from typing import List, Dict, Tuple
from langchain_core.documents import Document

class ContextAugmenter:
    """
    Implements Prev-Next Augmentation.
    It takes retrieved chunks and attempts to fetch their neighbors 
    from the original document corpus to restore context.
    """

    def __init__(self, all_chunks: List[Document]):
        # We build a fast lookup map: (source, chunk_index) -> Document
        self.lookup_map = {}
        for doc in all_chunks:
            source = doc.metadata.get("source", "unknown")
            idx = doc.metadata.get("chunk_index", -1)
            if idx != -1:
                self.lookup_map[(source, idx)] = doc

    def augment(self, retrieved_docs: List[Document], window_size: int = 1) -> List[Document]:
        """
        For each retrieved doc, grab 'window_size' neighbors before and after.
        Merges them into a single expanded document.
        """
        if window_size == 0:
            return retrieved_docs

        augmented_docs = []

        for doc in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            center_idx = doc.metadata.get("chunk_index", -1)
            
            if center_idx == -1:
                # No metadata, return as is
                augmented_docs.append(doc)
                continue

            # Gather indices: [center-1, center, center+1]
            indices_to_fetch = range(center_idx - window_size, center_idx + window_size + 1)
            
            text_parts = []
            
            for i in indices_to_fetch:
                # Check if this neighbor exists in the map
                if (source, i) in self.lookup_map:
                    text_parts.append(self.lookup_map[(source, i)].page_content)
            
            # Create a new "Expanded" document
            new_content = "\n".join(text_parts)
            
            new_doc = Document(
                page_content=new_content,
                metadata=doc.metadata.copy()
            )
            # Mark it as augmented
            new_doc.metadata["augmented"] = True
            new_doc.metadata["original_size"] = len(doc.page_content)
            new_doc.metadata["augmented_size"] = len(new_content)
            
            augmented_docs.append(new_doc)

        return augmented_docs