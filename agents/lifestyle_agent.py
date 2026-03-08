"""
Lifestyle & Wellness Agent
Handles questions about diet, exercise, sleep, stress management, and healthy living.
Provides evidence-based lifestyle recommendations.
"""

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate


class LifestyleAgent:
    def __init__(self, retriever, llm_config=None):
        """
        Initialize Lifestyle & Wellness Agent
        
        Args:
            retriever: Pinecone retriever for medical/wellness documents
            llm_config: Configuration for the LLM
        """
        self.retriever = retriever
        
        # Configure LLM
        if llm_config is None:
            llm_config = {
                'temperature': 0.5,  # Slightly higher for more natural lifestyle advice
                'max_tokens': 600
            }
        
        self.llm = OpenAI(**llm_config)
        
        # Lifestyle-specific prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized Lifestyle & Wellness Advisor.
Your expertise includes:
- Nutrition and diet recommendations (especially for PCOS and hormonal health)
- Exercise and physical activity guidance
- Sleep hygiene and stress management
- Healthy lifestyle habits
- Preventive health measures

Provide practical, evidence-based advice that's:
- Actionable and specific
- Safe and sustainable
- Tailored to women's health when relevant
- Backed by the provided medical context

Always emphasize the importance of consulting healthcare providers or registered dietitians
for personalized medical nutrition therapy or treatment plans.

Context:
{context}"""),
            ("human", "{question}")
        ])
    
    def process(self, question):
        """
        Process lifestyle and wellness questions
        
        Args:
            question: User's lifestyle-related question
            
        Returns:
            str: Lifestyle recommendations and advice
        """
        # Retrieve relevant wellness documents
        docs = self.retriever.invoke(question)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate lifestyle advice with string prompt
        prompt_text = f"""You are a Lifestyle & Wellness Advisor specializing in women's health.

Provide practical, evidence-based advice with:

**Key Recommendations:**
- Use bullet points (with -) for specific, actionable steps
- Make advice clear and easy to follow
- Focus on sustainable, safe practices

**Important Notes:**
- Mention benefits of each recommendation
- Emphasize consulting healthcare providers for personalized plans

Context:
{context}

Question: {question}

Advice:"""
        
        response = self.llm.invoke(prompt_text)
        
        return response.strip()
    
    def get_agent_type(self):
        """Return the agent type identifier"""
        return "lifestyle"
