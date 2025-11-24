import os
import torch
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingProvider:
    """
    Factory to return the specific embedding model based on configuration.
    """
    
    def __init__(self):
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def get_embedding_model(self, model_name: str = "openai", model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}

        # Set device in kwargs if not present
        if "device" not in model_kwargs:
            model_kwargs["device"] = self.device

        if model_name == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                print("Warning: OPENAI_API_KEY not found.")
            return OpenAIEmbeddings()
            
        else:
            print(f"Loading Embedding Model on {self.device.upper()}: {model_name}...")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs
            )