import os
import sys
import shutil
import copy
import itertools
import argparse
from typing import List, Dict

from config_schema import RAGConfig
from ingest.loaders import DocumentLoader
from ingest.chunkers import Chunker
from evaluation.synthetic import SyntheticDataGenerator
from optimizer.pipeline_runner import PipelineRunner
from exporter.deploy_builder import DeployBuilder

# --- USER CONFIG ---
DATA_DIR = "my_data"
EXPORT_DIR = "best_rag_pipeline"
GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
EVAL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# --- EXPANDED SEARCH SPACE ---
CHUNK_SIZES = [256, 512, 1024]
EMBEDDINGS = ["all-MiniLM-L6-v2"] 
RETRIEVERS = ["dense", "hybrid", "sparse"] # Added Sparse
TOP_K_OPTIONS = [3, 5]

RERANKERS = [None, "cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]

# Feature 1: Context Window
# 0 = Standard chunk. 1 = Chunk + Prev + Next.
CONTEXT_WINDOWS = [0, 1]

PROMPTS = ["default", "concise"]

def setup_data_and_test_set():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        sys.exit(1)
    loader = DocumentLoader()
    raw_docs = loader.load_directory(DATA_DIR)
    if not raw_docs:
        sys.exit(1)
    
    print(f"[+] Loaded {len(raw_docs)} documents.")
    
    # Chunk specifically for generation
    gen_chunker = Chunker(chunk_size=500, chunk_overlap=0)
    gen_chunks = gen_chunker.split(raw_docs)
    
    qa_gen = SyntheticDataGenerator(generator_llm_name=GENERATOR_MODEL)
    test_set = qa_gen.generate_qa_pairs(gen_chunks, num_questions=5)
    
    if not test_set:
        test_set = [{"question": "Summarize?", "ground_truth": "Content."}]
    return test_set

def run_sequential_search(runner: PipelineRunner) -> RAGConfig:
    print(f"\n[MODE] SEQUENTIAL SEARCH ACTIVATED (Advanced)")
    
    # 0. Baseline
    current_best = RAGConfig(
        chunk_size=256,
        chunk_overlap=30,
        embedding_model=EMBEDDINGS[0],
        retriever_method="dense",
        top_k=3,
        context_window=0,
        reranker_model=None,
        llm_model=EVAL_MODEL,
        prompt_style="default"
    )
    print(f"[+] Baseline: {current_best}")

    # Phase 1: Chunk Size
    print("\n--- Phase 1: Chunk Size ---")
    configs = []
    for opt in CHUNK_SIZES:
        c = copy.copy(current_best)
        c.chunk_size = opt
        c.chunk_overlap = int(opt * 0.1)
        configs.append(c)
    result = runner.run_experiment(configs)
    current_best = result['config']
    print(f">> Winner Phase 1: Chunk={current_best.chunk_size} (Score: {result['score']:.4f})")

    # Phase 2: Retriever Method
    print("\n--- Phase 2: Retriever Method ---")
    configs = []
    for opt in RETRIEVERS:
        c = copy.copy(current_best)
        c.retriever_method = opt
        configs.append(c)
    result = runner.run_experiment(configs)
    current_best = result['config']
    print(f">> Winner Phase 2: Method={current_best.retriever_method} (Score: {result['score']:.4f})")

    # Phase 3: Reranking Model (Feature 3)
    print("\n--- Phase 3: Reranker Model ---")
    configs = []
    for opt in RERANKERS:
        c = copy.copy(current_best)
        c.reranker_model = opt
        configs.append(c)
    result = runner.run_experiment(configs)
    current_best = result['config']
    print(f">> Winner Phase 3: Reranker={current_best.reranker_model} (Score: {result['score']:.4f})")

    # Phase 4: Context Window (Feature 1)
    print("\n--- Phase 4: Prev-Next Augmentation ---")
    configs = []
    for opt in CONTEXT_WINDOWS:
        c = copy.copy(current_best)
        c.context_window = opt
        configs.append(c)
    result = runner.run_experiment(configs)
    current_best = result['config']
    print(f">> Winner Phase 4: Window={current_best.context_window} (Score: {result['score']:.4f})")

    return current_best, result['score']

def run_grid_search(runner: PipelineRunner) -> RAGConfig:
    print(f"\n[MODE] GRID SEARCH ACTIVATED")
    configs = []
    # Exhaustive combination
    for chunk, ret, rerank, win, prompt in itertools.product(
        CHUNK_SIZES, RETRIEVERS, RERANKERS, CONTEXT_WINDOWS, PROMPTS
    ):
        cfg = RAGConfig(
            chunk_size=chunk, chunk_overlap=int(chunk*0.1),
            embedding_model=EMBEDDINGS[0],
            retriever_method=ret, top_k=3,
            context_window=win, reranker_model=rerank,
            llm_model=EVAL_MODEL, prompt_style=prompt
        )
        configs.append(cfg)
    
    print(f"[+] Testing {len(configs)} configs...")
    res = runner.run_experiment(configs)
    return res['config'], res['score']

def export_pipeline(config: RAGConfig):
    print(f"\n[+] Exporting to '{EXPORT_DIR}/'...")
    builder = DeployBuilder()
    builder.export(config, output_dir=EXPORT_DIR)
    dest_doc_dir = os.path.join(EXPORT_DIR, "documents")
    if os.path.exists(dest_doc_dir):
        shutil.rmtree(dest_doc_dir)
    shutil.copytree(DATA_DIR, dest_doc_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential", "grid"], default="sequential")
    args = parser.parse_args()

    test_set = setup_data_and_test_set()
    runner = PipelineRunner(documents_dir=DATA_DIR, test_set=test_set)

    if args.mode == "grid":
        winner, score = run_grid_search(runner)
    else:
        winner, score = run_sequential_search(runner)

    print(f"\nWINNING CONFIG (Score: {score:.4f}):\n{winner}")
    export_pipeline(winner)

if __name__ == "__main__":
    main()