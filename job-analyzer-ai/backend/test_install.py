# backend/test_install.py
# Run this to make sure everything installed correctly

print("Testing installations...")

import fastapi
print(f"✅ FastAPI      : {fastapi.__version__}")

import chromadb
print(f"✅ ChromaDB     : {chromadb.__version__}")

from sentence_transformers import SentenceTransformer
print(f"✅ SentenceTransformers : imported")

import groq
print(f"✅ Groq         : imported")

import PyPDF2
print(f"✅ PyPDF2       : imported")

# Quick embedding test — make sure it actually works
print("\n🔄 Loading embedding model (downloads ~90MB first time)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
test  = model.encode(["Hello this is a test"])
print(f"✅ Embedding model works! Vector size: {test.shape}")

# Quick ChromaDB test
print("\n🔄 Testing ChromaDB...")
client     = chromadb.Client()
collection = client.create_collection("test")
collection.add(documents=["hello world"], ids=["1"])
results    = collection.query(query_texts=["hello"], n_results=1)
print(f"✅ ChromaDB works! Found: {results['documents'][0][0]}")

print("\n🎉 ALL GOOD! Ready to build.")