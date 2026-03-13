# backend/main.py
#
# THIS FILE = THE SERVER
# Connects everything together:
# Frontend → FastAPI → RAG Engine → Analyzer → Groq → Response
#
# ENDPOINTS:
# POST /analyze     → submit JD → get full analysis
# POST /chat        → ask question about JD
# GET  /health      → check server is running

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os

try:
    from .rag_engine import index_jd
    from .analyzer import run_full_analysis, chat_with_jd
except ImportError:
    from rag_engine import index_jd
    from analyzer import run_full_analysis, chat_with_jd

# ============================================
# CREATE FASTAPI APP
# ============================================
app = FastAPI(
    title       = "Job Description Analyzer AI",
    description = "Paste any JD → get skill gap, roadmap, interview questions, resume tips",
    version     = "1.0.0"
)

# ============================================
# CORS — Allow frontend to talk to backend
# Without this browser blocks the request
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ============================================
# INPUT SCHEMAS
# Pydantic validates incoming data automatically
# If frontend sends wrong data → auto error
# ============================================

class JDRequest(BaseModel):
    jd_text: str           # the job description text

class ChatRequest(BaseModel):
    question     : str     # user's question
    collection_id: str     # which JD to search in


# ============================================
# ROUTE 1: Health Check
# GET /health
# Just to confirm server is running
# ============================================
@app.get("/health")
def health_check():
    return {
        "status" : "running",
        "model"  : "llama-3.3-70b-versatile",
        "version": "1.0.0"
    }


# ============================================
# ROUTE 2: Analyze JD
# POST /analyze
#
# FLOW:
# 1. Receive JD text from frontend
# 2. Index it into ChromaDB (RAG)
# 3. Run all 4 analyses using Groq
# 4. Return everything to frontend
#
# This is the MAIN endpoint
# ============================================
@app.post("/analyze")
def analyze_jd(request: JDRequest):
    """
    Full JD analysis — skill gap, roadmap,
    interview questions, resume tips.
    """

    # Validate — make sure JD is not empty
    if not request.jd_text.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "JD text cannot be empty"
        )

    # Minimum length check
    if len(request.jd_text.split()) < 30:
        raise HTTPException(
            status_code = 400,
            detail      = "JD too short. Please paste the complete job description."
        )

    try:
        # STEP 1: Index JD into ChromaDB
        print("\nIndexing JD...")
        index_result  = index_jd(request.jd_text)
        collection_id = index_result["collection_id"]
        print(f"Indexed. Chunks: {index_result['total_chunks']}")

        # STEP 2: Run all 4 analyses
        print("Running full analysis...")
        analysis = run_full_analysis(collection_id)

        # STEP 3: Return everything
        return {
            "status"        : "success",
            "collection_id" : collection_id,   # frontend saves this for chat
            "total_chunks"  : index_result["total_chunks"],
            "total_words"   : index_result["total_words"],
            "analysis"      : analysis
        }

    except ValueError as e:
        # Our own validation errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        print(f"Error: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Analysis failed: {str(e)}"
        )


# ============================================
# ROUTE 3: Chat with JD
# POST /chat
#
# FLOW:
# 1. Receive question + collection_id
# 2. Retrieve relevant chunks from ChromaDB
# 3. Send to Groq with context
# 4. Return answer
# ============================================
@app.post("/chat")
def chat(request: ChatRequest):
    """
    Answer any question about the JD.
    """

    # Validate
    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Question cannot be empty"
        )

    if not request.collection_id.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Please analyze a JD first before chatting."
        )

    try:
        result = chat_with_jd(
            question      = request.question,
            collection_id = request.collection_id
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Chat failed: {str(e)}"
        )


# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run(
        "backend.main:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = reload_enabled       # auto restart on code change
    )
