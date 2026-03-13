# Job Analyzer AI

A minimal FastAPI + HTML UI skeleton for a Job Description Analyzer using a RAG-style pipeline.

## Quickstart

1. Create a `.env` file with your `GROQ_API_KEY` (already included as a placeholder).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the backend:

```bash
uvicorn backend.main:app --reload
```

4. Open the UI:

Open `frontend/index.html` in your browser.

## Project Structure

```
job-analyzer-ai/
+-- backend/
¦   +-- main.py
¦   +-- analyzer.py
¦   +-- embeddings.py
¦   +-- prompts.py
+-- frontend/
¦   +-- index.html
+-- .env
+-- requirements.txt
+-- README.md
```
