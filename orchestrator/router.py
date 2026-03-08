"""
Question Router
Routes user questions to the appropriate specialized agent based on content analysis.
"""

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import re


class QuestionRouter:
    def __init__(self, medical_agent, symptom_agent, lifestyle_agent, llm_config=None):
        """
        Initialize Question Router
        
        Args:
            medical_agent: MedicalAgent instance
            symptom_agent: SymptomAgent instance
            lifestyle_agent: LifestyleAgent instance
            llm_config: Configuration for routing LLM (optional)
        """
        self.medical_agent = medical_agent
        self.symptom_agent = symptom_agent
        self.lifestyle_agent = lifestyle_agent
        
        # Configure routing LLM (lighter config for classification)
        if llm_config is None:
            llm_config = {
                'temperature': 0.1,  # Low temperature for consistent routing
                'max_tokens': 50
            }
        
        self.routing_llm = OpenAI(**llm_config)
        
        # Routing prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question classifier for a health assistant.
Classify the question into ONE of these categories:

1. MEDICAL: Questions about diseases, conditions (PCOS, diabetes, etc.), 
   hormones, medical terminology, pathophysiology, or medical explanations.
   Examples: "What is PCOS?", "How does insulin resistance work?", "Explain endometriosis"

2. SYMPTOM: Questions describing symptoms or asking about potential causes of symptoms.
   Examples: "I have acne and irregular periods", "What causes fatigue?", "Why am I experiencing pain?"

3. LIFESTYLE: Questions about diet, nutrition, exercise, sleep, stress management, or wellness.
   Examples: "What diet helps PCOS?", "Best exercises for weight loss?", "How to reduce stress?"

Respond with ONLY ONE WORD: MEDICAL, SYMPTOM, or LIFESTYLE"""),
            ("human", "{question}")
        ])
    
    def _keyword_based_routing(self, question):
        """
        Simple keyword-based routing as fallback
        
        Args:
            question: User's question (lowercase)
            
        Returns:
            str: Agent type (medical, symptom, lifestyle)
        """
        question_lower = question.lower()
        
        # Symptom keywords
        symptom_keywords = [
            'symptom', 'pain', 'ache', 'feel', 'experiencing', 'have',
            'irregular', 'cramp', 'bleeding', 'discharge', 'swelling',
            'fatigue', 'tired', 'nausea', 'headache', 'dizzy'
        ]
        
        # Lifestyle keywords
        lifestyle_keywords = [
            'diet', 'food', 'eat', 'nutrition', 'meal', 'recipe',
            'exercise', 'workout', 'fitness', 'yoga', 'gym',
            'sleep', 'stress', 'relax', 'meditation', 'wellness',
            'weight', 'lose', 'gain', 'healthy', 'lifestyle'
        ]
        
        # Medical keywords
        medical_keywords = [
            'disease', 'condition', 'disorder', 'syndrome',
            'pcos', 'diabetes', 'thyroid', 'hormone', 'endocrine',
            'treatment', 'medication', 'diagnosis', 'what is',
            'explain', 'causes', 'pathology', 'mechanism'
        ]
        
        # Count keyword matches
        symptom_score = sum(1 for kw in symptom_keywords if kw in question_lower)
        lifestyle_score = sum(1 for kw in lifestyle_keywords if kw in question_lower)
        medical_score = sum(1 for kw in medical_keywords if kw in question_lower)
        
        # Determine routing based on scores
        scores = {
            'symptom': symptom_score,
            'lifestyle': lifestyle_score,
            'medical': medical_score
        }
        
        # Return agent with highest score, default to medical
        max_score = max(scores.values())
        if max_score == 0:
            return 'medical'  # Default to medical if no keywords match
        
        return max(scores, key=scores.get)
    
    def _llm_based_routing(self, question):
        """
        Use LLM to classify the question
        
        Args:
            question: User's question
            
        Returns:
            str: Agent type (medical, symptom, lifestyle)
        """
        try:
            formatted_prompt = self.routing_prompt.format_messages(question=question)
            response = self.routing_llm.invoke(formatted_prompt)
            
            # Parse response
            response_text = response.strip().upper()
            
            if 'SYMPTOM' in response_text:
                return 'symptom'
            elif 'LIFESTYLE' in response_text:
                return 'lifestyle'
            elif 'MEDICAL' in response_text:
                return 'medical'
            else:
                # Fallback to keyword-based
                return self._keyword_based_routing(question)
                
        except Exception as e:
            print(f"LLM routing failed: {e}. Using keyword-based routing.")
            return self._keyword_based_routing(question)
    
    def route_question(self, question):
        """
        Route question to appropriate agent and get response
        
        Args:
            question: User's question
            
        Returns:
            dict: Response with answer and metadata
        """
        # Determine which agent to use
        agent_type = self._llm_based_routing(question)
        
        print(f"🔀 Routing to: {agent_type.upper()} Agent")
        
        # Route to appropriate agent
        if agent_type == 'symptom':
            response = self.symptom_agent.process(question)
            agent_name = "Symptom Checker"
        elif agent_type == 'lifestyle':
            response = self.lifestyle_agent.process(question)
            agent_name = "Lifestyle & Wellness"
        else:  # medical
            response = self.medical_agent.process(question)
            agent_name = "Medical Knowledge"
        
        return {
            'answer': response,
            'agent_type': agent_type,
            'agent_name': agent_name
        }
    
    def route_with_enhanced_symptom_check(self, question):
        """
        Route question with enhanced symptom checking 
        (uses medical agent for additional context when symptoms detected)
        
        Args:
            question: User's question
            
        Returns:
            dict: Response with answer and metadata
        """
        agent_type = self._llm_based_routing(question)
        
        print(f"🔀 Routing to: {agent_type.upper()} Agent")
        
        if agent_type == 'symptom':
            # Enhanced symptom check with medical agent collaboration
            result = self.symptom_agent.analyze_with_medical_agent(question)
            response = f"{result['symptom_analysis']}\n\nMedical Context:\n{result['medical_explanation']}"
            agent_name = "Symptom Checker + Medical Knowledge"
        elif agent_type == 'lifestyle':
            response = self.lifestyle_agent.process(question)
            agent_name = "Lifestyle & Wellness"
        else:  # medical
            response = self.medical_agent.process(question)
            agent_name = "Medical Knowledge"
        
        return {
            'answer': response,
            'agent_type': agent_type,
            'agent_name': agent_name
        }
