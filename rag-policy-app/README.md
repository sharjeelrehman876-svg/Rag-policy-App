# TechCorp Policy Assistant — RAG LLM Application

A Retrieval-Augmented Generation (RAG) chatbot that answers employee questions about TechCorp's company policies, powered by **Groq LLM** (free tier), **HuggingFace sentence-transformers** (local embeddings), and **ChromaDB** (local vector store).

---

## 🏗️ Architecture Overview

```
User Question
     │
     ▼
Flask Web App (/chat endpoint)
     │
     ▼
Embed Question → sentence-transformers (local)
     │
     ▼
Vector Search → ChromaDB (Top-5 chunks)
     │
     ▼
Prompt Assembly (retrieved context + guardrails)
     │
     ▼
Groq LLM (llama3-8b-8192) → Answer + Citations
     │
     ▼
JSON Response → Web UI
```

---

## 📋 Policy Corpus

10 synthetic company policy documents totaling ~120 pages:

| Doc ID | Title |
|--------|-------|
| HR-001 | Paid Time Off (PTO) Policy |
| HR-002 | Remote Work Policy |
| HR-003 | Code of Conduct Policy |
| HR-004 | Performance Review Policy |
| HR-005 | Holiday and Leave Policy |
| HR-006 | Employee Onboarding Policy |
| HR-007 | Employee Benefits Policy |
| FIN-001 | Expense Reimbursement Policy |
| IT-001 | Information Security Policy |
| IT-002 | Data Privacy Policy |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.11+
- Git
- A free [Groq API key](https://console.groq.com) (takes 2 minutes to get)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-policy-app.git
cd rag-policy-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

> **Get your free Groq API key:** https://console.groq.com  
> Sign up → API Keys → Create API Key (free tier: 14,400 req/day)

### 5. Run the Application

```bash
python app.py
```

The app will:
1. Automatically parse and index all policy documents (first run takes ~30 seconds to embed)
2. Start the Flask server at `http://localhost:5000`

Open your browser at **http://localhost:5000**

---

## 🚀 API Endpoints

### `GET /`
Web chat interface.

### `POST /chat`
Ask a policy question.

**Request:**
```json
{
  "question": "How many PTO days do I get in my first year?"
}
```

**Response:**
```json
{
  "answer": "New employees receive 15 PTO days in their first year... [Source: HR-001 - Paid Time Off Policy]",
  "citations": [
    { "doc_id": "HR-001", "title": "Paid Time Off (PTO) Policy", "filename": "pto_policy.md" }
  ],
  "retrieved_chunks": [...],
  "latency_ms": 842.3
}
```

### `GET /health`
Health check.

```json
{ "status": "ok", "service": "TechCorp Policy RAG", "version": "1.0.0" }
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Running Evaluation

Make sure the app is running, then:

```bash
python evaluate.py
```

Results will be printed to the console and saved to `evaluation_results.json`.

---

## 🐳 Optional: Docker

```bash
docker build -t rag-policy-app .
docker run -p 5000:5000 -e GROQ_API_KEY=your_key rag-policy-app
```

---

## 📁 Project Structure

```
rag-policy-app/
├── app.py                  # Flask web application
├── rag_pipeline.py         # Core RAG pipeline (ingest, embed, retrieve, generate)
├── evaluate.py             # Evaluation script (20 questions)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
├── design-and-evaluation.md
├── ai-tooling.md
├── policies/               # 10 policy markdown documents
│   ├── pto_policy.md
│   ├── remote_work_policy.md
│   ├── expense_policy.md
│   ├── security_policy.md
│   ├── holiday_leave_policy.md
│   ├── code_of_conduct.md
│   ├── performance_review_policy.md
│   ├── onboarding_policy.md
│   ├── data_privacy_policy.md
│   └── benefits_policy.md
├── templates/
│   └── index.html          # Web chat UI
├── tests/
│   └── test_app.py         # pytest test suite
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI/CD
```

---

## 🔒 Guardrails

The RAG system enforces:
- **Out-of-scope refusal:** Refuses to answer questions not covered by the policy corpus
- **Output length limits:** Max 600 tokens per response
- **Citation enforcement:** Every answer must cite source document IDs
- **Low temperature (0.1):** Minimizes hallucination

---

## 📜 License

MIT License — TechCorp Policy RAG Application, 2024.
