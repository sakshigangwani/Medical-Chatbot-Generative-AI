"""
Medical Knowledge Agent
Handles questions about diseases, PCOS, hormones, and medical explanations.
Uses Pinecone RAG for retrieving medical information.
"""

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate


class MedicalAgent:
    def __init__(self, retriever, llm_config=None):
        """
        Initialize Medical Agent
        
        Args:
            retriever: Pinecone retriever for medical documents
            llm_config: Configuration for the LLM (temperature, max_tokens, etc.)
        """
        self.retriever = retriever
        
        # Configure LLM
        if llm_config is None:
            llm_config = {
                'temperature': 0.4,
                'max_tokens': 500
            }
        
        self.llm = OpenAI(**llm_config)
        
        # Medical-specific prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized Medical Knowledge Assistant.
Your expertise includes:
- Diseases and medical conditions (especially PCOS, hormonal disorders)
- Medical terminology and explanations
- Pathophysiology and disease mechanisms
- Hormonal health and endocrine system

Use the provided medical context to give accurate, evidence-based answers.
If you don't know something, clearly state your limitations.
Always recommend consulting healthcare professionals for diagnosis and treatment.
Keep responses clear, concise, and informative (3-5 sentences).

Context:
{context}"""),
            ("human", "{question}")
        ])
    
    def process(self, question):
        """
        Process a medical knowledge question
        
        Args:
            question: User's medical question
            
        Returns:
            str: Medical agent's response
        """
        # Retrieve relevant medical documents
        docs = self.retriever.invoke(question)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using LLM with string prompt
        prompt_text = f"""You are a Medical Knowledge Assistant.

Use the provided medical context to give accurate, evidence-based answers.
Format your response with:
- Clear paragraphs for main explanations
- Bullet points (use -) for lists of symptoms, treatments, or key points
- Keep responses informative but concise

Always recommend consulting healthcare professionals for diagnosis and treatment.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt_text)
        
        return response.strip()
    
    def get_agent_type(self):
        """Return the agent type identifier"""
        return "medical"
