
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Learning Assistant")


# Define the structure of incoming questions
class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Learning Assistant is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(data: Question):
    return {
        "question_received": data.question,
        "answer": "Your AI assistant will answer this soon!"
    }
