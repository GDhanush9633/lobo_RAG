import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv 
from fastapi.middleware.cors import CORSMiddleware
from app.rag import answer_question

# Load env vars locally (Azure ignores safely)
load_dotenv()

app = FastAPI(
    title="LOBO RAG POC",
    description="RAG POC using Azure OpenAI + Azure AI Search",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For POC only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Request schema
# ---------------------------
class QueryRequest(BaseModel):
    question: str

# ---------------------------
# Health check (Azure-friendly)
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# RAG endpoint
# ---------------------------
@app.post("/ask")
def ask_rag(req: QueryRequest):
    answer = answer_question(req.question)

    return {
        "question": req.question,
        "answer": answer,
    }
