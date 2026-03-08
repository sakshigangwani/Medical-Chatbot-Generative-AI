"""
Simple examples demonstrating how to use each agent directly
"""

from retrieval import MedicalRetriever
from agents import MedicalAgent, SymptomAgent, LifestyleAgent


def example_medical_agent():
    """Example: Using the Medical Knowledge Agent"""
    print("\n" + "=" * 70)
    print("🩺 Medical Knowledge Agent Example")
    print("=" * 70)
    
    # Initialize retriever and agent
    retriever = MedicalRetriever().get_medical_retriever()
    medical_agent = MedicalAgent(retriever)
    
    # Ask medical questions
    questions = [
        "What is PCOS?",
        "How does insulin resistance develop?",
        "Explain polycystic ovary syndrome"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        response = medical_agent.process(q)
        print(f"💬 Answer: {response}\n")


def example_symptom_agent():
    """Example: Using the Symptom Checker Agent"""
    print("\n" + "=" * 70)
    print("🔍 Symptom Checker Agent Example")
    print("=" * 70)
    
    # Initialize retriever and agents
    retriever = MedicalRetriever()
    medical_agent = MedicalAgent(retriever.get_medical_retriever())
    symptom_agent = SymptomAgent(
        retriever.get_symptom_retriever(),
        medical_agent
    )
    
    # Ask symptom-related questions
    questions = [
        "I have irregular periods and acne",
        "What causes excessive hair growth and weight gain?",
    ]
    
    for q in questions:
        print(f"\n❓ Symptoms: {q}")
        response = symptom_agent.process(q)
        print(f"💬 Analysis: {response}\n")


def example_lifestyle_agent():
    """Example: Using the Lifestyle & Wellness Agent"""
    print("\n" + "=" * 70)
    print("🌱 Lifestyle & Wellness Agent Example")
    print("=" * 70)
    
    # Initialize retriever and agent
    retriever = MedicalRetriever().get_lifestyle_retriever()
    lifestyle_agent = LifestyleAgent(retriever)
    
    # Ask lifestyle questions
    questions = [
        "What diet is recommended for PCOS?",
        "What exercises help with hormonal balance?",
        "How can I improve my sleep quality?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        response = lifestyle_agent.process(q)
        print(f"💬 Advice: {response}\n")


def example_router():
    """Example: Using the Question Router"""
    print("\n" + "=" * 70)
    print("🔀 Question Router Example (Automatic Routing)")
    print("=" * 70)
    
    from orchestrator import QuestionRouter
    
    # Initialize all components
    retriever = MedicalRetriever()
    
    medical_agent = MedicalAgent(retriever.get_medical_retriever())
    symptom_agent = SymptomAgent(
        retriever.get_symptom_retriever(),
        medical_agent
    )
    lifestyle_agent = LifestyleAgent(retriever.get_lifestyle_retriever())
    
    router = QuestionRouter(medical_agent, symptom_agent, lifestyle_agent)
    
    # Ask mixed questions - router will automatically route to correct agent
    questions = [
        "What is PCOS?",  # → Medical
        "I have acne and irregular periods",  # → Symptom
        "What diet helps manage PCOS?",  # → Lifestyle
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        result = router.route_question(q)
        print(f"🤖 Routed to: {result['agent_name']}")
        print(f"💬 Response: {result['answer']}\n")


if __name__ == "__main__":
    print("🚀 FlowSync Multi-Agent System - Usage Examples")
    print("=" * 70)
    
    # Run examples
    example_medical_agent()
    example_symptom_agent()
    example_lifestyle_agent()
    example_router()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
