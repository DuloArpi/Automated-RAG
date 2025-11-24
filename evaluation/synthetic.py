import random
import re
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from generation.llm import LLMFactory

class SyntheticDataGenerator:
    def __init__(self, generator_llm_name="gpt-4o-mini"):
        self.llm = LLMFactory.create_llm(generator_llm_name)

    def generate_qa_pairs(self, documents: List[Document], num_questions: int = 5) -> List[Dict]:
        qa_pairs = []
        
        # Simplified prompt for small models
        prompt = PromptTemplate.from_template(
            """Context:
            {text}
            
            Task: Write one simple question about the text above, and then write the answer.
            Use this exact format:
            
            Q: [The question]
            A: [The answer]
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()

        selected_docs = documents
        if len(documents) > num_questions:
            selected_docs = random.sample(documents, num_questions)

        print(f"Generating {len(selected_docs)} QA pairs...")

        for doc in selected_docs:
            try:
                output = chain.invoke({"text": doc.page_content})
                
                # Robust Parsing with Regex
                # Looks for Q: ... A: ... (multiline supported)
                match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", output, re.DOTALL | re.IGNORECASE)
                
                if match:
                    q_part = match.group(1).strip()
                    a_part = match.group(2).strip()
                    
                    qa_pairs.append({
                        "question": q_part,
                        "ground_truth": a_part,
                        "context_source": doc.page_content
                    })
                else:
                    # Fallback for messy output
                    if "?" in output:
                        parts = output.split("?")
                        qa_pairs.append({
                            "question": parts[0].strip() + "?",
                            "ground_truth": parts[1].strip()[:100], # Take first 100 chars of rest
                            "context_source": doc.page_content
                        })

            except Exception as e:
                print(f"Error generating QA: {e}")

        return qa_pairs