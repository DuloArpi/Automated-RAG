import time
import copy
from typing import List, Dict
from tqdm import tqdm

from config_schema import RAGConfig
from ingest.loaders import DocumentLoader
from ingest.chunkers import Chunker
from indexing.embeddings import EmbeddingProvider
from indexing.vector_store import VectorStoreManager
from retrieval.retrievers import RetrieverFactory
from retrieval.rerankers import Reranker
from retrieval.augmentation import ContextAugmenter
from generation.llm import LLMFactory
from generation.prompts import PromptFactory
from evaluation.metrics import Evaluator
from langchain_core.output_parsers import StrOutputParser

class PipelineRunner:
    def __init__(self, documents_dir: str, test_set: List[Dict] = None):
        self.loader = DocumentLoader()
        self.raw_docs = self.loader.load_directory(documents_dir)
        self.test_set = test_set
        self.results = []

        if not self.raw_docs:
            raise ValueError(f"No documents found in {documents_dir}")
        
        print(f"Loaded {len(self.raw_docs)} source documents.")

    def run_experiment(self, configs: List[RAGConfig]):
        print(f"Starting Experiment with {len(configs)} configurations...")
        
        for config in tqdm(configs, desc="Evaluating Pipelines"):
            try:
                score, details = self._evaluate_single_config(config)
                
                result = {
                    "score": score,
                    "config": config,
                    "details": details
                }
                self.results.append(result)
                # print(f"  > Score: {score:.4f} | {config}")
                
            except Exception as e:
                print(f"  > Failed config {config}: {e}")
                # raise e # Uncomment to debug crashes

        self.results.sort(key=lambda x: x["score"], reverse=True)
        return self.results[0]

    def _evaluate_single_config(self, config: RAGConfig):
        # 1. Chunking
        chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = chunker.split(self.raw_docs)
        
        # 2. Embeddings & Indexing
        embed_provider = EmbeddingProvider()
        embed_model = embed_provider.get_embedding_model(config.embedding_model)
        
        vs_manager = VectorStoreManager(embed_model)
        vector_store = vs_manager.create_vector_store(chunks, collection_name=None)
        
        # 3. Retriever
        # Note: If augmentation is on, we might want to retrieve slightly more docs first
        # to ensure we have enough candidates before reranking/augmenting.
        fetch_k = config.top_k * 2 if config.reranker_model else config.top_k

        retriever = RetrieverFactory.create_retriever(
            vector_store=vector_store,
            documents=chunks,
            method=config.retriever_method,
            k=fetch_k 
        )
        
        # 4. Components: Augmenter & Reranker
        augmenter = ContextAugmenter(all_chunks=chunks)
        
        reranker = None
        if config.reranker_model:
            reranker = Reranker(model_name=config.reranker_model)
            
        # 5. LLM
        llm = LLMFactory.create_llm(config.llm_model)
        prompt_template = PromptFactory.get_prompt(config.prompt_style)
        
        # 6. Evaluation Loop
        total_score = 0
        num_samples = len(self.test_set)
        
        last_answer = ""
        last_metrics = {}

        for i, sample in enumerate(self.test_set):
            q = sample["question"]
            truth = sample["ground_truth"]
            
            # A. Retrieve
            docs = retriever.invoke(q)
            
            # B. Rerank (Before augmentation usually works best to filter noise)
            if reranker:
                docs = reranker.rerank(q, docs, top_n=config.top_k)
            else:
                docs = docs[:config.top_k]

            # C. Augment (Feature 1: Prev-Next)
            if config.context_window > 0:
                docs = augmenter.augment(docs, window_size=config.context_window)
            
            # D. Generate
            context_str = "\n\n".join([d.page_content for d in docs])
            chain = prompt_template | llm | StrOutputParser()
            
            try:
                answer = chain.invoke({"context": context_str, "question": q})
            except:
                answer = ""
            
            # E. Score
            metrics = Evaluator.calculate_metrics(answer, truth)
            total_score += metrics["f1_score"]
            
            if i == 0:
                last_answer = answer
                last_metrics = metrics
            
        avg_score = total_score / num_samples if num_samples > 0 else 0
        return avg_score, {"metrics": last_metrics, "last_answer": last_answer}