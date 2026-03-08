# FlowSync Multi-Agent Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                                                                   │
│  ┌────────────────────────┐      ┌──────────────────────────┐  │
│  │   Web Chat (Flask)     │      │   API Endpoints (JSON)   │  │
│  │   http://localhost:8080│      │   /api/chat, /api/agents │  │
│  └────────────┬───────────┘      └────────────┬─────────────┘  │
└───────────────┼──────────────────────────────┼─────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK APPLICATION                           │
│                          (app.py)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR LAYER                             │
│                   (orchestrator/router.py)                       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Question Router (Hybrid)                     │  │
│  │  • LLM-based classification (OpenAI)                      │  │
│  │  • Keyword-based fallback                                 │  │
│  │  • Routing logic & decision engine                        │  │
│  └───┬──────────────────┬──────────────────┬────────────────┘  │
└──────┼──────────────────┼──────────────────┼────────────────────┘
       │                  │                  │
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Medical    │  │   Symptom    │  │  Lifestyle   │
│    Agent     │  │   Checker    │  │    Agent     │
│     🩺       │  │     🔍       │  │     🌱       │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                  │                  │
       │  agents/         │  agents/         │  agents/
       │  medical_agent   │  symptom_agent   │  lifestyle_agent
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                              │
│                  (retrieval/retriever.py)                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Medical Retriever (MedicalRetriever)            │  │
│  │  • Manages Pinecone connections                           │  │
│  │  • Optimized retrievers for each agent                    │  │
│  │  • Similarity search with scores                          │  │
│  └───────────────────────┬──────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Pinecone Vector Store (medical-chatbot)           │  │
│  │  • Vector embeddings (HuggingFace)                        │  │
│  │  • Medical document chunks                                │  │
│  │  • Semantic search enabled                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Source Documents (Data/)                     │  │
│  │  • PDF medical documents                                  │  │
│  │  • PCOS, hormones, health information                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Specialization

```
┌─────────────────────────────────────────────────────────────────┐
│                  🩺 MEDICAL KNOWLEDGE AGENT                      │
├─────────────────────────────────────────────────────────────────┤
│ Expertise:                                                       │
│ • Diseases & Conditions (PCOS, diabetes, thyroid)               │
│ • Hormonal Health & Endocrine System                            │
│ • Medical Terminology & Explanations                            │
│ • Pathophysiology & Disease Mechanisms                          │
├─────────────────────────────────────────────────────────────────┤
│ Configuration:                                                   │
│ • Temperature: 0.4 (balanced)                                   │
│ • Max Tokens: 500                                               │
│ • Retrieval K: 3 documents                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  🔍 SYMPTOM CHECKER AGENT                        │
├─────────────────────────────────────────────────────────────────┤
│ Expertise:                                                       │
│ • Symptom Interpretation & Analysis                             │
│ • Pattern Recognition in Symptom Clusters                       │
│ • Preliminary Condition Assessment                              │
│ • Severity Evaluation                                           │
├─────────────────────────────────────────────────────────────────┤
│ Configuration:                                                   │
│ • Temperature: 0.3 (consistent)                                 │
│ • Max Tokens: 600                                               │
│ • Retrieval K: 4 documents (more context)                       │
│ • Collaboration: Can call Medical Agent                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                🌱 LIFESTYLE & WELLNESS AGENT                     │
├─────────────────────────────────────────────────────────────────┤
│ Expertise:                                                       │
│ • Nutrition & Diet Recommendations                              │
│ • Exercise & Physical Activity Guidance                         │
│ • Sleep Hygiene & Stress Management                             │
│ • Preventive Health Measures                                    │
├─────────────────────────────────────────────────────────────────┤
│ Configuration:                                                   │
│ • Temperature: 0.5 (creative)                                   │
│ • Max Tokens: 600                                               │
│ • Retrieval K: 3 documents                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

```
1. User Question
   "What diet helps with PCOS?"
          │
          ▼
2. Flask App (app.py)
   Receives request
          │
          ▼
3. Question Router (orchestrator/router.py)
   ┌─────────────────────────────────┐
   │ LLM Classification               │
   │ "diet" → LIFESTYLE              │
   └─────────────┬───────────────────┘
                 │
                 ▼
4. Lifestyle Agent (agents/lifestyle_agent.py)
   ┌─────────────────────────────────┐
   │ Process question                 │
   │ Get context from retriever       │
   │ Generate response with LLM       │
   └─────────────┬───────────────────┘
                 │
                 ▼
5. Medical Retriever (retrieval/retriever.py)
   ┌─────────────────────────────────┐
   │ Query Pinecone                   │
   │ Get top 3 relevant docs          │
   │ Return context                   │
   └─────────────┬───────────────────┘
                 │
                 ▼
6. Response Generation
   ┌─────────────────────────────────┐
   │ Combine context + prompt         │
   │ Generate with OpenAI             │
   │ Return formatted response        │
   └─────────────┬───────────────────┘
                 │
                 ▼
7. Return to User
   {
     "answer": "For PCOS, focus on...",
     "agent_type": "lifestyle",
     "agent_name": "Lifestyle & Wellness Agent"
   }
```

## Routing Decision Tree

```
                    User Question
                         │
                         ▼
            ┌─────────────────────────┐
            │  Question Router         │
            │  (LLM + Keywords)        │
            └──────┬───────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   Medical?    Symptom?   Lifestyle?
        │          │          │
        │          │          │
   ┌────┴───┐ ┌───┴────┐ ┌──┴─────┐
   │Medical │ │Symptom │ │Lifestyle│
   │Keywords│ │Keywords│ │Keywords│
   └────┬───┘ └───┬────┘ └──┬─────┘
        │          │          │
   disease     symptom      diet
   condition    pain       exercise
   pcos         ache       sleep
   hormone      feel       stress
   explain      have       nutrition
        │          │          │
        └──────────┴──────────┘
                   │
                   ▼
            Route to Agent
```

## Data Flow

```
┌─────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ PDF     │──1──▶│ Document │──2──▶│ Chunks & │──3──▶│ Pinecone │
│ Files   │      │ Loader   │      │ Embeddings│      │ Storage  │
└─────────┘      └──────────┘      └──────────┘      └──────────┘
   Data/           helper.py         HuggingFace       Vector DB
                                                            │
                                                            │
                                                            ▼
┌─────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│  User   │◀─6───│  Agent   │◀─5───│ Retriever│◀─4───│ Similarity│
│Response │      │ Response │      │  Context │      │  Search   │
└─────────┘      └──────────┘      └──────────┘      └──────────┘
   JSON            OpenAI LLM        retriever.py     Pinecone
```

## Component Dependencies

```
app.py
  ├── orchestrator/router.py
  │     ├── agents/medical_agent.py
  │     │     └── retrieval/retriever.py
  │     ├── agents/symptom_agent.py
  │     │     ├── retrieval/retriever.py
  │     │     └── agents/medical_agent.py (collaboration)
  │     └── agents/lifestyle_agent.py
  │           └── retrieval/retriever.py
  │
  └── retrieval/retriever.py
        ├── src/helper.py (embeddings)
        └── PineconeVectorStore
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND                                   │
│  • HTML/CSS (templates/chat.html, static/styles.css)           │
│  • JavaScript (in chat.html)                                    │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND                                    │
│  • Flask (Web Framework)                                        │
│  • Python 3.8+                                                  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                       AI/ML                                      │
│  • LangChain (Framework)                                        │
│  • OpenAI GPT (LLM)                                             │
│  • HuggingFace (Embeddings)                                     │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE                               │
│  • Pinecone (Vector Store)                                      │
│  • Semantic Search                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

**Legend:**
- 🩺 Medical Knowledge Agent
- 🔍 Symptom Checker Agent
- 🌱 Lifestyle & Wellness Agent
- 🔀 Question Router
- 📚 Retrieval System
- 📊 Vector Database
