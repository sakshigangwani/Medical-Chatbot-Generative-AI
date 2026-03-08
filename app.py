from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# Import multi-agent system components
from retrieval import MedicalRetriever
from agents import MedicalAgent, SymptomAgent, LifestyleAgent
from orchestrator import QuestionRouter

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Initialize retriever
print("📚 Initializing Medical Retriever...")
medical_retriever = MedicalRetriever(index_name="medical-chatbot", search_k=3)

# Get retrievers for different agents
base_retriever = medical_retriever.get_retriever()
symptom_retriever = medical_retriever.get_symptom_retriever()
lifestyle_retriever = medical_retriever.get_lifestyle_retriever()

# Initialize specialized agents
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

# Initialize router/orchestrator
print("🔀 Initializing Question Router...")
router = QuestionRouter(
    medical_agent=medical_agent,
    symptom_agent=symptom_agent,
    lifestyle_agent=lifestyle_agent
)

print("✅ Multi-Agent System Ready!")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"\n❓ Question: {msg}")

    # Route question to appropriate agent
    result = router.route_question(msg)
    
    response = result['answer']
    agent_type = result['agent_type']
    agent_name = result['agent_name']
    
    print(f"🤖 Agent: {agent_name}")
    print(f"💬 Response: {response[:100]}...")

    # Return response with agent information
    return str(response)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    JSON API endpoint for chat with agent metadata
    Returns: JSON with answer, agent_type, and agent_name
    """
    data = request.get_json()
    msg = data.get("message", "")
    
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    print(f"\n❓ API Question: {msg}")
    
    # Route question to appropriate agent
    result = router.route_question(msg)
    
    print(f"🤖 Agent: {result['agent_name']}")
    
    return jsonify({
        "answer": result['answer'],
        "agent_type": result['agent_type'],
        "agent_name": result['agent_name']
    })


@app.route("/api/agents", methods=["GET"])
def get_agents_info():
    """
    Get information about available agents
    """
    return jsonify({
        "agents": [
            {
                "type": "medical",
                "name": "Medical Knowledge Agent",
                "description": "Handles diseases, PCOS, hormones, and medical explanations",
                "expertise": ["Diseases", "PCOS", "Hormones", "Medical terminology", "Pathophysiology"]
            },
            {
                "type": "symptom",
                "name": "Symptom Checker Agent",
                "description": "Analyzes symptoms and suggests possible conditions",
                "expertise": ["Symptom interpretation", "Pattern recognition", "Preliminary assessment"]
            },
            {
                "type": "lifestyle",
                "name": "Lifestyle & Wellness Agent",
                "description": "Provides advice on diet, exercise, sleep, and stress",
                "expertise": ["Nutrition", "Exercise", "Sleep hygiene", "Stress management", "Wellness"]
            }
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
