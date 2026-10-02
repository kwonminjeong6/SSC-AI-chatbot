from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag

app = FastAPI(
    title="University AI Chatbot",
    description="대학 공식문서 기반 고성능 RAG 챗봇 (경진대회용)",
    version="2.0.0"
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"message": "University AI Chatbot is running smoothly."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag(request.question)
    return result