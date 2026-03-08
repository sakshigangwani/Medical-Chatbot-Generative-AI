"""
Test script for FlowSync Multi-Agent System
Tests routing to different specialized agents
"""

from retrieval import MedicalRetriever
from agents import MedicalAgent, SymptomAgent, LifestyleAgent
from orchestrator import QuestionRouter


def test_multi_agent_system():
    """Test the multi-agent system with sample questions"""
    
    print("=" * 70)
    print("🧪 FlowSync Multi-Agent System Test")
    print("=" * 70)
    
    # Initialize system
    print("\n📚 Initializing Medical Retriever...")
    medical_retriever = MedicalRetriever(index_name="medical-chatbot", search_k=3)
    
    base_retriever = medical_retriever.get_retriever()
    symptom_retriever = medical_retriever.get_symptom_retriever()
    lifestyle_retriever = medical_retriever.get_lifestyle_retriever()
    
    print("🩺 Initializing Medical Knowledge Agent...")
    medical_agent = MedicalAgent(
        retriever=base_retriever,
        llm_config={'temperature': 0.4, 'max_tokens': 500}
    )
    
    print("🔍 Initializing Symptom Checker Agent...")
    symptom_agent = SymptomAgent(
        retriever=symptom_retriever,
        medical_agent=medical_agent,
        llm_config={'temperature': 0.3, 'max_tokens': 600}
    )
    
    print("🌱 Initializing Lifestyle & Wellness Agent...")
    lifestyle_agent = LifestyleAgent(
        retriever=lifestyle_retriever,
        llm_config={'temperature': 0.5, 'max_tokens': 600}
    )
    
    print("🔀 Initializing Question Router...")
    router = QuestionRouter(
        medical_agent=medical_agent,
        symptom_agent=symptom_agent,
        lifestyle_agent=lifestyle_agent
    )
    
    print("✅ Multi-Agent System Ready!\n")
    
    # Test questions
    test_questions = [
        {
            "question": "What is PCOS and how does it affect the body?",
            "expected_agent": "medical"
        },
        {
            "question": "I have irregular periods and acne. What could be wrong?",
            "expected_agent": "symptom"
        },
        {
            "question": "What diet is recommended for managing PCOS?",
            "expected_agent": "lifestyle"
        },
        {
            "question": "Explain how insulin resistance develops",
            "expected_agent": "medical"
        },
        {
            "question": "What exercises help with weight loss in PCOS?",
            "expected_agent": "lifestyle"
        }
    ]
    
    # Run tests
    print("\n" + "=" * 70)
    print("🧪 Running Test Cases")
    print("=" * 70 + "\n")
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}/{len(test_questions)}")
        print(f"{'─' * 70}")
        print(f"❓ Question: {test['question']}")
        print(f"🎯 Expected Agent: {test['expected_agent'].upper()}")
        
        # Get response
        result = router.route_question(test['question'])
        
        print(f"🤖 Actual Agent: {result['agent_type'].upper()} ({result['agent_name']})")
        
        # Check if routing is correct
        if result['agent_type'] == test['expected_agent']:
            print("✅ Routing: CORRECT")
        else:
            print("⚠️  Routing: UNEXPECTED (but may be reasonable)")
        
        print(f"\n💬 Response:\n{result['answer']}\n")
    
    print("=" * 70)
    print("🎉 Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_multi_agent_system()
