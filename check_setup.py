"""
Simple connectivity test for FlowSync Multi-Agent System
"""
import os
from dotenv import load_dotenv

print("=" * 70)
print("🔍 Testing FlowSync Setup")
print("=" * 70)

# Test 1: Environment variables
print("\n1️⃣ Checking environment variables...")
load_dotenv()

pinecone_key = os.environ.get('PINECONE_API_KEY')
openai_key = os.environ.get('OPENAI_API_KEY')

if pinecone_key:
    print(f"   ✅ PINECONE_API_KEY: {'*' * 10}{pinecone_key[-4:]}")
else:
    print("   ❌ PINECONE_API_KEY not found in .env")

if openai_key:
    print(f"   ✅ OPENAI_API_KEY: {'*' * 10}{openai_key[-4:]}")
else:
    print("   ❌ OPENAI_API_KEY not found in .env")

# Test 2: Package imports
print("\n2️⃣ Testing package imports...")
try:
    from retrieval import MedicalRetriever
    print("   ✅ retrieval module")
except Exception as e:
    print(f"   ❌ retrieval module: {e}")

try:
    from agents import MedicalAgent, SymptomAgent, LifestyleAgent
    print("   ✅ agents module")
except Exception as e:
    print(f"   ❌ agents module: {e}")

try:
    from orchestrator import QuestionRouter
    print("   ✅ orchestrator module")
except Exception as e:
    print(f"   ❌ orchestrator module: {e}")

# Test 3: Core dependencies
print("\n3️⃣ Testing core dependencies...")
dependencies = [
    'langchain',
    'langchain_openai',
    'langchain_pinecone',
    'flask',
    'pinecone',
    'sentence_transformers'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✅ {dep}")
    except ImportError:
        print(f"   ❌ {dep} - run: pip install {dep}")

print("\n" + "=" * 70)
print("✅ Setup Check Complete!")
print("=" * 70)

if pinecone_key and openai_key:
    print("\n🚀 Ready to test! Try:")
    print("   • python quick_test.py       (quick verification)")
    print("   • python app.py              (start web server)")
    print("   • Visit: http://localhost:8080")
else:
    print("\n⚠️  Add missing API keys to .env file before testing")
