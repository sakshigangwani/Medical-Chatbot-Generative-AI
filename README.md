# 🩺 FlowSync AI Health Assistant

Multi-agent AI health assistant for women’s health support using Retrieval-Augmented Generation (RAG).

## Overview

FlowSync AI Health Assistant routes each user query to the most suitable specialist agent:

- `MedicalAgent` for medical concepts, conditions, and explanations
- `SymptomAgent` for symptom interpretation and next-step guidance
- `LifestyleAgent` for wellness, nutrition, sleep, stress, and habits

The system combines:

- OpenAI LLM responses
- Pinecone vector retrieval
- HuggingFace embeddings
- Flask web app + chat UI

---

## Key Features

- Multi-agent architecture with intelligent routing
- RAG-backed responses grounded in uploaded health documents
- Dedicated API endpoints for chat and system checks
- Clean UI with response formatting and typing animation
- Modular, testable Python codebase

---

## Architecture

User Query → `QuestionRouter` → Specialist Agent → Retriever (Pinecone) → LLM Response → UI/API

Core modules:

- `agents/` — agent implementations
- `orchestrator/router.py` — routing logic
- `retrieval/retriever.py` — retriever setup/invocation
- `app.py` — Flask app and endpoints
- `store_index.py` — vector indexing pipeline

---

## Project Structure

```text
flowsync-ai-health-assistant/
├── agents/
│   ├── medical_agent.py
│   ├── symptom_agent.py
│   └── lifestyle_agent.py
├── orchestrator/
│   └── router.py
├── retrieval/
│   └── retriever.py
├── src/
│   ├── helper.py
│   └── prompt.py
├── templates/
│   └── chat.html
├── static/
│   └── styles.css
├── Data/
├── app.py
├── store_index.py
├── check_setup.py
├── quick_test.py
├── test_agents.py
├── examples.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1) Clone

```bash
git clone https://github.com/sakshigangwani/flowsync-ai-health-assistant.git
cd flowsync-ai-health-assistant
```

### 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add environment variables

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 5) Verify setup

```bash
python check_setup.py
```

---

## Run

### Build / refresh vector index

```bash
python store_index.py
```

### Start app

```bash
python app.py
```

Open: http://127.0.0.1:8080

---

## API

### Chat

`POST /api/chat`

Example request:

```json
{
	"message": "What is PCOS?"
}
```

Example response:

```json
{
	"answer": "...",
	"agent_type": "medical",
	"agent_name": "Medical Knowledge Agent"
}
```

### Agent listing

`GET /api/agents`

---

## Testing

- Quick smoke test: `python quick_test.py`
- Agent-level tests: `python test_agents.py`
- Example flows: `python examples.py`

---

## Notes

- This assistant provides informational support only.
- It is not a diagnosis tool and does not replace professional medical care.
- In emergencies, contact local emergency services immediately.

---

## Author

Sakshi Gangwani  
MS Computer Science, University of Southern California
