import os
import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMFactory:
    """
    Returns an LLM backend based on configuration, with GPU support.
    """

    @staticmethod
    def create_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        
        # 1. OpenAI Models
        if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
            if "OPENAI_API_KEY" not in os.environ:
                print("Warning: OPENAI_API_KEY not set. OpenAI calls will fail.")
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature
            )

        # 2. Local Models (HuggingFace)
        else:
            # Detect Device
            if torch.cuda.is_available():
                device_type = "cuda"
                # Use float16 on GPU for speed and memory efficiency
                dtype = torch.float16 
                print(f"Loading Local LLM on GPU (CUDA): {model_name}...")
            elif torch.backends.mps.is_available():
                device_type = "mps"
                dtype = torch.float16
                print(f"Loading Local LLM on Apple Silicon (MPS): {model_name}...")
            else:
                device_type = "cpu"
                dtype = torch.float32
                print(f"Loading Local LLM on CPU (Slow): {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load Model with device_map="auto" (Requires 'accelerate' library)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="auto" if device_type == "cuda" else None, # auto handles multi-gpu
            )

            # Move to MPS explicitly if needed (auto doesn't always support MPS perfect)
            if device_type == "mps":
                model.to("mps")
            elif device_type == "cpu":
                model.to("cpu")

            # GPT-2 / Llama pad token fix
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

            # Create Pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                temperature=temperature if temperature > 0 else 0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                # Note: We don't pass 'device' here because 'model' is already on device
            )

            return HuggingFacePipeline(pipeline=pipe)