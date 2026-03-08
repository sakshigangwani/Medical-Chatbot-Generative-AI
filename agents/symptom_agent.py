"""
Symptom Checker Agent
Handles symptom interpretation and suggests possible conditions.
Works in collaboration with the Medical Agent for detailed explanations.
"""

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate


class SymptomAgent:
    def __init__(self, retriever, medical_agent, llm_config=None):
        """
        Initialize Symptom Checker Agent
        
        Args:
            retriever: Pinecone retriever for medical documents
            medical_agent: Reference to Medical Agent for detailed explanations
            llm_config: Configuration for the LLM
        """
        self.retriever = retriever
        self.medical_agent = medical_agent
        
        # Configure LLM
        if llm_config is None:
            llm_config = {
                'temperature': 0.3,  # Lower temperature for more consistent symptom analysis
                'max_tokens': 600
            }
        
        self.llm = OpenAI(**llm_config)
        
        # Symptom analysis prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized Symptom Checker Assistant.
Your role is to:
- Analyze symptoms described by users
- Identify patterns and potential conditions
- Provide preliminary assessments based on medical knowledge
- Suggest when to seek professional medical help

IMPORTANT: You are NOT providing diagnosis, only helping understand symptoms.
Always recommend consulting healthcare professionals for proper diagnosis.

Analyze the symptoms and relevant medical context to provide:
1. Possible conditions that match these symptoms
2. Why these symptoms might occur
3. Severity assessment (mild, moderate, urgent)
4. Recommendation for next steps

Context:
{context}"""),
            ("human", "{symptoms}")
        ])
    
    def process(self, question):
        """
        Process symptom-related questions
        
        Args:
            question: User's symptom description
            
        Returns:
            str: Symptom analysis and recommendations
        """
        # Retrieve relevant medical documents about symptoms
        docs = self.retriever.invoke(question)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate symptom analysis with string prompt
        prompt_text = f"""You are a Symptom Checker Assistant.

IMPORTANT: You are NOT providing diagnosis, only helping understand symptoms.

Analyze the symptoms and provide:

**Possible Conditions:**
- List potential conditions (use bullet points with -)

**Why These Symptoms Occur:**
- Explain the underlying causes

**Recommended Next Steps:**
- Suggest when to seek professional help
- Provide severity assessment

Always recommend consulting healthcare professionals for proper diagnosis.

Context:
{context}

Symptoms: {question}

Analysis:"""
        
        response = self.llm.invoke(prompt_text)
        
        return response.strip()
    
    def analyze_with_medical_agent(self, question):
        """
        Analyze symptoms and get detailed medical explanation
        
        Args:
            question: User's symptom description
            
        Returns:
            dict: Combined symptom analysis and medical explanation
        """
        # Get symptom analysis
        symptom_analysis = self.process(question)
        
        # Extract potential condition and ask medical agent for details
        # This provides a more comprehensive response
        medical_context = self.medical_agent.process(
            f"Explain the medical background for: {question}"
        )
        
        return {
            "symptom_analysis": symptom_analysis,
            "medical_explanation": medical_context,
            "agent_type": "symptom"
        }
    
    def get_agent_type(self):
        """Return the agent type identifier"""
        return "symptom"
