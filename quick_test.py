"""
Quick Test Script for Multi-Agent System
Tests the agents without running Flask server
"""

print("=" * 70)
print("🧪 Quick Test - FlowSync Multi-Agent System")
print("=" * 70)

# Test 1: Test imports
print("\n✅ Testing imports...")
try:
    from retrieval import MedicalRetriever
    from agents import MedicalAgent, SymptomAgent, LifestyleAgent
    from orchestrator import QuestionRouter
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Initialize components
print("\n✅ Testing component initialization...")
try:
    print("  📚 Initializing Medical Retriever...")
    medical_retriever = MedicalRetriever(index_name="medical-chatbot", search_k=3)
    
    base_retriever = medical_retriever.get_retriever()
    symptom_retriever = medical_retriever.get_symptom_retriever()
    lifestyle_retriever = medical_retriever.get_lifestyle_retriever()
    
    print("  🩺 Initializing Medical Knowledge Agent...")
    medical_agent = MedicalAgent(
        retriever=base_retriever,
        llm_config={'temperature': 0.4, 'max_tokens': 500}
    )
    
    print("  🔍 Initializing Symptom Checker Agent...")
    symptom_agent = SymptomAgent(
        retriever=symptom_retriever,
        medical_agent=medical_agent,
        llm_config={'temperature': 0.3, 'max_tokens': 600}
    )
    
    print("  🌱 Initializing Lifestyle & Wellness Agent...")
    lifestyle_agent = LifestyleAgent(
        retriever=lifestyle_retriever,
        llm_config={'temperature': 0.5, 'max_tokens': 600}
    )
    
    print("  🔀 Initializing Question Router...")
    router = QuestionRouter(
        medical_agent=medical_agent,
        symptom_agent=symptom_agent,
        lifestyle_agent=lifestyle_agent
    )
    
    print("✓ All components initialized successfully!")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Test routing
print("\n✅ Testing question routing...")
test_questions = [
    ("What is PCOS?", "medical"),
    ("I have irregular periods", "symptom"),
    ("What diet helps PCOS?", "lifestyle"),
]

for question, expected_type in test_questions:
    try:
        print(f"\n  Question: '{question}'")
        result = router.route_question(question)
        print(f"  → Routed to: {result['agent_name']} ({result['agent_type']})")
        
        if result['agent_type'] == expected_type:
            print(f"  ✓ Correct routing!")
        else:
            print(f"  ⚠ Expected {expected_type}, got {result['agent_type']}")
        
        # Show first 150 characters of response
        print(f"  Response preview: {result['answer'][:150]}...")
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("🎉 Quick Test Complete!")
print("=" * 70)
print("\n💡 Next Steps:")
print("   1. Run full tests: python test_agents.py")
print("   2. Run examples: python examples.py")
print("   3. Start web app: python app.py")
print("   4. Visit: http://localhost:8080")
