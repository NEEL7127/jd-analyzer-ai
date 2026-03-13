🚀 What It Does
Most people waste hours manually reading job descriptions. ChatGPT gives generic advice that has nothing to do with your specific JD.
JDAnalyzer.ai reads YOUR exact JD using a RAG pipeline and gives you:
OutputWhat You Get🎯 Skill Gap AnalysisMust-have vs nice-to-have skills from YOUR JD📚 30-Day RoadmapDay-by-day prep plan based on THIS specific role❓ Interview QuestionsPredicted questions extracted from YOUR JD📝 Resume TipsExact ATS keywords, headline, bullet suggestions💬 Chat with JDAsk anything — answers sourced from JD only

⚡ Performance
Before:  4 sequential Groq API calls  =  ~20 seconds
After:   4 parallel calls via ThreadPoolExecutor  =  ~8 seconds
Result:  60% faster response time

🏗️ Architecture — RAG Pipeline
User pastes JD
      ↓
✂️  PARSE      — JD chunked into 300-token overlapping segments
      ↓
🧠  EMBED      — Each chunk → 384-dim vector (all-MiniLM-L6-v2)
      ↓
🗄️  STORE      — Vectors stored in ChromaDB (in-memory)
      ↓
🔍  RETRIEVE   — Top 3-4 relevant chunks fetched per query
      ↓
⚡  GENERATE   — Llama 3.3 70B via Groq — 4 analyses run in parallel
      ↓
✅  RESULTS    — Skill gap + Roadmap + Interview Qs + Resume Tips
Why RAG instead of just prompting?

Grounds every answer in YOUR specific JD — not generic knowledge
Prevents hallucination — model only sees retrieved content
Works for any JD, any domain, any company


🛠️ Tech Stack
LayerTechnologyWhyBackendFastAPI (Python)Async, fast, production-readyVector DBChromaDBIn-memory vector store, zero setupEmbeddingsMiniLM-L6-v2384-dim, CPU-only, completely freeLLMLlama 3.3 70BBest open-source reasoning modelInferenceGroq API500+ tokens/sec, free tierParallelismThreadPoolExecutor4 simultaneous Groq callsFrontendHTML + CSS + JSSingle file, no framework needed

📁 Project Structure
jd-analyzer-ai/
├── backend/
│   ├── __init__.py
│   ├── main.py           — FastAPI server, /analyze and /chat endpoints
│   ├── analyzer.py       — Parallel Groq calls, 4 analyses at once
│   └── rag_engine.py     — Chunking, embedding, ChromaDB retrieval
├── frontend/
│   └── index.html        — Complete single-file frontend
├── .env                  — GROQ_API_KEY (not committed to git)
├── .gitignore
├── requirements.txt
└── README.md

⚙️ Local Setup
Step 1 — Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/jd-analyzer-ai.git
cd jd-analyzer-ai
Step 2 — Install dependencies
bashpip install -r requirements.txt
Step 3 — Add your Groq API key
bashecho "GROQ_API_KEY=your_key_here" > .env
Get a free API key at: console.groq.com
Step 4 — Run the backend
bashuvicorn backend.main:app --reload
Step 5 — Open the frontend
Open frontend/index.html in your browser


🔌 API Endpoints
POST /analyze
Analyzes a job description — returns all 4 results.
Request:
json{
  "jd_text": "Full job description text here..."
}
Response:
json{
  "collection_id": "jd_abc123",
  "analysis": {
    "skill_analysis": "...",
    "learning_roadmap": "...",
    "interview_questions": "...",
    "resume_tips": "..."
  },
  "total_words": 420,
  "total_chunks": 8
}
POST /chat
Chat with the JD using RAG.
Request:
json{
  "question": "Is this a senior or junior role?",
  "collection_id": "jd_abc123"
}
Response:
json{
  "answer": "Based on the JD, this appears to be a mid-level role..."
}
GET /health
json{ "status": "ok" }

🌐 Deployment
Backend → Render.com (free tier)
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.main:app --host 0.0.0.0 --port 10000
Env Variable:  GROQ_API_KEY = your_key
Frontend → Netlify (free tier)
Drag and drop the frontend/ folder onto netlify.com
Done — live URL in 30 seconds

💡 Key Technical Decisions
1. Parallel API calls — 60% speed gain
python# Sequential (slow — ~20s):
result1 = call_groq(prompt1)
result2 = call_groq(prompt2)
result3 = call_groq(prompt3)
result4 = call_groq(prompt4)

# Parallel (fast — ~8s):
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(call_groq, p): name for name, p in prompts}
2. Chunking with overlap — no information loss
python# 50-token overlap prevents losing context at chunk boundaries
def chunk_text(text, chunk_size=300, overlap=50)
3. RAG over fine-tuning — right choice for this use case
Fine-tuning: expensive, needs GPU, static knowledge
RAG:         dynamic, free, adapts to any new JD instantly

🎓 What This Project Demonstrates
✅ RAG Pipeline        — real architecture, not just API wrapper
✅ Vector Databases    — ChromaDB, semantic embeddings, cosine similarity
✅ LLM Integration     — Groq API, Llama 3.3 70B, prompt engineering
✅ Optimization        — ThreadPoolExecutor, 60% speed improvement
✅ Production Backend  — FastAPI, CORS, error handling, REST design
✅ Full Stack          — end-to-end product from scratch
✅ Product Thinking    — real use case, real problem, real users

👤 Built By
Neel Deshmane
Diploma Computer Engineering · Pune, Maharashtra 🇮🇳
Aspiring AI/ML Engineer
