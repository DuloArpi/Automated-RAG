from langchain_core.prompts import PromptTemplate

class PromptFactory:
    
    STYLES = {
        "default": """You are a helpful assistant. Use the context below to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""",
        
        "concise": """Answer the question strictly based on the context. Be concise.
        Context: {context}
        Question: {question}
        Answer:""",
        
        "reasoning": """Analyze the context provided and answer the question step-by-step.
        Context: {context}
        Question: {question}
        Reasoning & Answer:"""
    }

    @staticmethod
    def get_prompt(style: str = "default") -> PromptTemplate:
        template_str = PromptFactory.STYLES.get(style, PromptFactory.STYLES["default"])
        return PromptTemplate.from_template(template_str)