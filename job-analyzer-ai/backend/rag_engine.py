# backend/rag_engine.py
# 
# THIS FILE = THE ENTIRE RAG BRAIN
# 
# It does 3 jobs:
# Job 1 → Take JD text → chunk it → embed it → store in ChromaDB
# Job 2 → Take a question → find relevant chunks from ChromaDB
# Job 3 → Return those chunks to be sent to Groq
#
# Think of this file as a LIBRARIAN:
# - First time: reads the book, makes index cards (indexing)
# - Later: you ask a question, librarian finds the right cards (retrieval)

import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib

# ============================================
# LOAD EMBEDDING MODEL
# 
# all-MiniLM-L6-v2 = small, fast, FREE model
# Runs perfectly on your i5 CPU — no GPU needed
# Converts any text → 384 numbers (vector)
# Similar meaning = similar vectors
# Downloads ~90MB first time, then cached forever
# ============================================
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ============================================
# CHROMADB SETUP
#
# chromadb.Client() = in-memory database
# Data lives as long as server is running
# Perfect for our use case — each user's JD
# is temporary, no need to save permanently
# ============================================
chroma_client = chromadb.Client()


# ============================================
# HELPER: Generate unique ID for each JD
#
# Why? ChromaDB needs a collection name.
# Each user's JD gets its own collection.
# We hash the first 100 chars of JD to make
# a unique ID — so two different JDs never
# mix with each other.
# ============================================
def get_collection_id(jd_text: str) -> str:
    """
    Generate a unique collection ID from JD text.
    Uses MD5 hash of first 200 characters.
    """
    hash_input = jd_text[:200].encode('utf-8')
    return "jd_" + hashlib.md5(hash_input).hexdigest()[:12]
    # Example output: "jd_a3f8c2d1b9e4"


# ============================================
# JOB 1: CHUNKING
#
# BABY EXPLANATION:
# We can't embed the WHOLE JD as one piece —
# it's too long and we'd lose specific details.
# Instead we cut it into small overlapping pieces.
#
# WHY OVERLAPPING?
# Imagine JD text:
# "...Python developer with FastAPI experience.
#  Must know Docker and AWS. Salary: 8-12 LPA..."
#
# Chunk 1: "Python developer with FastAPI experience."
# Chunk 2: "Must know Docker and AWS. Salary: 8-12"
# 
# But what if "FastAPI experience. Must know Docker"
# is the most relevant part for a question?
# It got CUT between chunks!
#
# OVERLAP fixes this:
# Chunk 1: "Python developer with FastAPI experience. Must know"
# Chunk 2: "FastAPI experience. Must know Docker and AWS."
# Now the important bridge is captured in both ✅
# ============================================
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Parameters:
        text       : full JD text
        chunk_size : words per chunk (300 = ~2 paragraphs)
        overlap    : words shared between consecutive chunks
    
    Returns:
        list of text chunks
    """
    # Clean the text first
    # Remove extra whitespace and weird characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into individual words
    words = text.split()
    
    chunks  = []
    start   = 0
    
    while start < len(words):
        # Take chunk_size words starting from 'start'
        end   = start + chunk_size
        chunk = ' '.join(words[start:end])
        
        # Only add if chunk has meaningful content
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        
        # Move start forward by (chunk_size - overlap)
        # This creates the overlap with next chunk
        start += chunk_size - overlap
    
    return chunks


# ============================================
# JOB 2: INDEX THE JD
#
# This function:
# 1. Chunks the JD text
# 2. Converts each chunk to embedding vector
# 3. Stores everything in ChromaDB
#
# Called ONCE when user submits their JD
# ============================================
def index_jd(jd_text: str) -> dict:
    """
    Process and store a job description in ChromaDB.
    
    INPUT : raw JD text (pasted by user)
    OUTPUT: collection_id to use for querying later
    """
    
    # Validate input
    if not jd_text or len(jd_text.strip()) < 50:
        raise ValueError("JD text too short. Please paste the full job description.")
    
    # Generate unique ID for this JD
    collection_id = get_collection_id(jd_text)
    
    # Delete old collection with same ID if exists
    # (in case user re-submits same JD)
    try:
        chroma_client.delete_collection(collection_id)
    except:
        pass  # collection didn't exist, that's fine
    
    # Create fresh collection
    collection = chroma_client.create_collection(
        name     = collection_id,
        metadata = {"hnsw:space": "cosine"}
        # cosine = measure similarity by angle between vectors
        # better than euclidean distance for text
    )
    
    # STEP 1: Chunk the JD
    chunks = chunk_text(jd_text)
    
    if len(chunks) == 0:
        raise ValueError("Could not extract text from JD.")
    
    # STEP 2: Convert all chunks to embeddings at once
    # .encode() = run embedding model on each chunk
    # Returns numpy array of shape (num_chunks, 384)
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks)
    
    # STEP 3: Store in ChromaDB
    collection.add(
        documents  = chunks,
        # ChromaDB needs list of lists for embeddings
        embeddings = embeddings.tolist(),
        # Each chunk needs unique ID
        ids        = [f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Indexed {len(chunks)} chunks into ChromaDB")
    
    return {
        "collection_id" : collection_id,
        "total_chunks"  : len(chunks),
        "total_words"   : len(jd_text.split()),
        "status"        : "indexed"
    }


# ============================================
# JOB 3: RETRIEVE RELEVANT CHUNKS
#
# Given a question + collection_id:
# 1. Convert question to embedding
# 2. Search ChromaDB for similar chunks
# 3. Return top N most relevant chunks
#
# Called every time we want to ask something
# about the JD
# ============================================
def retrieve(query: str, collection_id: str, n_results: int = 3) -> str:
    """
    Find most relevant JD chunks for a given query.
    
    INPUT:
        query         : what we're looking for
        collection_id : which JD collection to search
        n_results     : how many chunks to return
    
    OUTPUT:
        combined text of most relevant chunks
    """
    
    # Get the collection
    try:
        collection = chroma_client.get_collection(collection_id)
    except:
        raise ValueError("JD not found. Please submit your JD again.")
    
    # Convert query to embedding
    query_embedding = embedder.encode([query]).tolist()
    
    # Search ChromaDB
    # This finds chunks whose vectors are most similar
    # to the query vector — by cosine similarity
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = min(n_results, collection.count())
        # min() = don't ask for more chunks than exist
    )
    
    # results['documents'] = [[chunk1, chunk2, chunk3]]
    # We want the inner list
    chunks = results['documents'][0]
    
    # Combine chunks into one context string
    # Numbered so LLM knows these are separate sections
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Section {i}]\n{chunk}\n\n"
    
    return context.strip()


# ============================================
# BONUS: Get full JD text back
# Used when we want to analyze the ENTIRE JD
# not just specific parts
# ============================================
def get_full_context(collection_id: str) -> str:
    """
    Retrieve ALL chunks from a collection.
    Used for full document analysis.
    """
    try:
        collection = chroma_client.get_collection(collection_id)
    except:
        raise ValueError("JD not found. Please submit your JD again.")
    
    # Get everything stored in this collection
    all_data = collection.get()
    chunks   = all_data['documents']
    
    # Join all chunks
    return "\n\n".join(chunks)


# ============================================
# TEST THIS FILE DIRECTLY
# Run: python backend/rag_engine.py
# ============================================
if __name__ == "__main__":
    
    # Sample JD for testing
    test_jd = """
    Job Title: AI/ML Engineer
    Company: TechCorp India, Pune
    
    About the Role:
    We are looking for a passionate AI/ML Engineer to join our growing 
    team. You will work on cutting-edge machine learning projects and 
    help build AI-powered products used by millions of users.
    
    Required Skills:
    - Python (3+ years experience)
    - Machine Learning (scikit-learn, XGBoost)
    - Deep Learning (PyTorch or TensorFlow)
    - FastAPI or Flask for model deployment
    - SQL and basic database knowledge
    - Git and version control
    
    Nice to Have:
    - Experience with LLMs and RAG systems
    - Docker and Kubernetes
    - AWS or Google Cloud
    - MLflow or similar MLOps tools
    
    Responsibilities:
    - Build and deploy ML models to production
    - Work with data engineers to build data pipelines
    - Collaborate with product team to define AI features
    - Write clean, documented, testable code
    - Participate in code reviews
    
    Qualifications:
    - B.Tech or M.Tech in Computer Science or related field
    - 1-3 years of industry experience (freshers with strong projects welcome)
    
    Compensation:
    - Salary: 8-18 LPA based on experience
    - Stock options available
    - Remote friendly (2 days office per week)
    
    Location: Pune, Maharashtra (Hybrid)
    """
    
    print("=" * 50)
    print("     RAG ENGINE TEST")
    print("=" * 50)
    
    # Test 1: Index the JD
    print("\nSTEP 1: Indexing JD...")
    result = index_jd(test_jd)
    print(f"Collection ID  : {result['collection_id']}")
    print(f"Total chunks   : {result['total_chunks']}")
    print(f"Total words    : {result['total_words']}")
    
    collection_id = result['collection_id']
    
    # Test 2: Retrieve for different queries
    print("\nSTEP 2: Testing retrieval...")
    
    queries = [
        "What Python skills are required?",
        "What is the salary for this job?",
        "Is this job remote or office?",
        "What ML frameworks do they need?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        context = retrieve(query, collection_id, n_results=2)
        # Show just first 150 chars of result
        print(f"Found: {context[:150]}...")
    
    print("\nRAG Engine working.")
    print("=" * 50)
