# AI Tooling Documentation

**Project:** TechCorp Policy Assistant — RAG LLM Application

---

## AI Tools Used

### 1. Claude (Anthropic) — Primary Development Assistant

**How it was used:**
- Designed the overall RAG architecture and technology stack choices
- Generated all 10 synthetic company policy documents with realistic HR/Finance/IT content
- Wrote the core RAG pipeline code (`rag_pipeline.py`) including chunking, embedding, retrieval, and generation logic
- Created the Flask application structure (`app.py`) with proper error handling and endpoint design
- Designed the web chat UI (`templates/index.html`) with a clean, professional interface
- Generated the GitHub Actions CI/CD workflow (`.github/workflows/ci.yml`)
- Created the pytest test suite (`tests/test_app.py`)
- Wrote the evaluation script (`evaluate.py`) with 20 policy questions
- Produced all documentation (`README.md`, `design-and-evaluation.md`, `ai-tooling.md`)

**What worked well:**
- Claude excelled at generating realistic, coherent policy documents with consistent formatting and cross-references (e.g., HR-001 references HR-005 for holidays)
- Very effective at producing clean, well-commented Python code on the first attempt
- Excellent at explaining design tradeoffs (e.g., why ChromaDB over Pinecone, why K=5 over K=3)
- Generated comprehensive test cases that covered both happy paths and edge cases (empty input, oversized input)

**What didn't work as well:**
- Needed multiple iterations to refine the system prompt to achieve both strong guardrails AND good answer quality simultaneously
- Initial chunk size suggestion (200 words) was too small for policy documents; had to revise to 400 words after testing

---

### 2. Groq Console (UI)

**How it was used:**
- Created the free Groq API key at console.groq.com
- Tested raw LLM prompts via the Groq playground before integrating into the codebase
- Monitored rate limit usage during evaluation runs

**What worked well:**
- The Groq playground made it very fast to iterate on the system prompt design
- Rate limit dashboard helped understand throughput constraints during evaluation

---

### 3. HuggingFace Hub (Model Discovery)

**How it was used:**
- Browsed the MTEB leaderboard to identify the best free embedding model for semantic similarity
- Selected `all-MiniLM-L6-v2` based on its benchmark scores and download count (indicating community validation)

**What worked well:**
- Clear benchmarks made the embedding model selection straightforward

---

## Summary

The majority of the code, documentation, and policy corpus was generated with Claude's assistance. Claude was used as a "senior AI engineer pair programmer" — providing architectural guidance, writing production-quality code, and explaining design rationale. Human oversight was applied throughout to validate correctness, test the application manually, and refine prompts based on real output quality.

The use of AI tooling significantly accelerated development — the entire project was scaffolded in approximately 4–6 hours rather than the estimated 2–3 days of manual development time.
