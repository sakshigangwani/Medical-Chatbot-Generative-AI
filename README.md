## 🩺 FlowSync AI Health Assistant

### Generative AI Medical Chatbot using Retrieval-Augmented Generation (RAG)

FlowSync AI Health Assistant is an AI-powered medical chatbot designed to answer questions related to **women’s health, cycles, hormones, PCOS, and general medical knowledge**.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from medical documents and generate reliable responses using Large Language Models.

This project demonstrates how to build a **production-style AI application using vector databases, embeddings, and conversational AI.**

---

## 🚀 Features

- AI-powered medical Q&A
- Retrieval-Augmented Generation (RAG) architecture
- Document-based medical knowledge retrieval
- Vector similarity search using Pinecone
- Real-time conversational chat interface
- Typing animation for AI responses
- Thinking animation while retrieving information
- Responsive chatbot UI
- Modular backend architecture

---

## 🧠 Architecture

This project follows the **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of allowing the LLM to hallucinate answers, the system retrieves relevant information from medical documents before generating responses.

### Workflow

1. Medical documents are loaded and split into smaller chunks.
2. Each chunk is converted into **vector embeddings**.
3. Embeddings are stored inside **Pinecone vector database**.
4. When a user asks a question:
   - The query is converted into an embedding.
   - Pinecone retrieves the most relevant document chunks.
5. Retrieved context + user question are sent to the LLM.
6. The LLM generates a contextual and accurate response.

---

## 🧩 Tech Stack

### Backend
- Python
- Flask

### AI / Machine Learning
- LangChain
- HuggingFace Embeddings
- Retrieval-Augmented Generation (RAG)

### Vector Database
- Pinecone

### Frontend
- HTML
- CSS
- JavaScript

### Other Tools
- python-dotenv (environment variable management)

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/sakshigangwani/flowsync-ai-health-assistant.git
cd flowsync-ai-health-assistant
```

---

### Create Virtual Environment

```bash
conda create -n medibot python=3.10
conda activate medibot
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
PINECONE_API_KEY=your_api_key
OPENAI_API_KEY=your_key
```

---

## 📚 Step 1 — Create Vector Database

Run the indexing script to convert documents into embeddings.

```bash
python store_index.py
```

This process will:

- Load medical PDF documents
- Split text into chunks
- Generate embeddings using HuggingFace
- Store embeddings in Pinecone vector database

---

## 💬 Step 2 — Run the Chatbot

Start the Flask application.

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:8080
```

---

## 🖥️ Chat Interface

The chatbot UI includes:

- FlowSync AI health assistant interface
- Typing animation for AI responses
- Thinking animation while retrieving knowledge
- Responsive chat design

Users can ask questions such as:

- What is PCOS?
- Why is my menstrual cycle irregular?
- Symptoms of hormonal imbalance
- What causes acne during periods?

---

## 📈 Future Improvements

Possible enhancements for this project:

- Multi-document knowledge base
- Medical conversation memory
- Personalized health insights
- Voice-based AI interaction
- LLM fine-tuning for medical accuracy
- Integration with the **FlowSync health platform**

---

## 👩‍💻 Author

**Sakshi Gangwani**

MS Computer Science  
University of Southern California

Focused on building **AI-powered health and productivity products**, including **FlowSync — a PCOS health management platform.**

---

## ⭐ Support

If you found this project useful, consider **starring the repository** on GitHub.
