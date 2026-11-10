
import os

from chunking import split_text
from fastapi import UploadFile, File, HTTPException
from pypdf import PdfReader
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# Initialize FastAPI
app = FastAPI(title="AI Learning Assistant")


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

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=data.question
    )

    return {
        "question": data.question,
        "answer": response.text
    }



@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    contents = await file.read()

    try:
        pdf = PdfReader(BytesIO(contents))

        extracted_text = ""

        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            extracted_text += f"\nPage {page_number + 1}:\n{text}"

        # Split extracted text into chunks
        chunks = split_text(
            extracted_text,
            chunk_size=500,
            overlap=50
        )

        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "total_characters": len(extracted_text),
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(error)}"
        )
