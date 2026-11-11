
import os
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

from chunking import split_text
from vector_store import store_chunks, search_chunks


# Load environment variables
load_dotenv()


# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# Initialize FastAPI
app = FastAPI(title="AI Learning Assistant")


# Request model
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


# General AI question endpoint

@app.post("/ask")
def ask_question(data: Question):

    # Step 1: Retrieve relevant chunks from ChromaDB
    results = search_chunks(data.question)

    if not results or not results["documents"][0]:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Upload a PDF first."
        )

    # Step 2: Prepare context for Gemini
    context_parts = []
    sources = []

    for i, document in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]

        context_parts.append(
            f"Source {i + 1}: {source}\n{document}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # Step 3: Create a RAG prompt
    prompt = f"""
    You are an AI Learning Assistant.

    Answer the user's question using ONLY the context
    provided below.

    If the answer is not available in the context,
    clearly say that the uploaded documents do not
    contain enough information.

    Do not invent facts.

    CONTEXT:
    {context}

    USER QUESTION:
    {data.question}

    Provide a clear and helpful answer.
    """

    # Step 4: Send context and question to Gemini
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    # Step 5: Return answer and sources
    return {
        "question": data.question,
        "answer": response.text,
        "sources": sources
    }


# Upload and process PDF
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    contents = await file.read()

    try:
        # Read PDF
        pdf = PdfReader(BytesIO(contents))

        # Extract text from every page
        extracted_text = ""

        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            extracted_text += (
                f"\nPage {page_number + 1}:\n{text}"
            )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF"
            )

        # Split extracted text into chunks
        chunks = split_text(
            extracted_text,
            chunk_size=500,
            overlap=50
        )

        # Store chunks and embeddings in ChromaDB
        stored_count = store_chunks(
            chunks,
            file.filename
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "total_characters": len(extracted_text),
            "total_chunks": len(chunks),
            "stored_chunks": stored_count
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF"
        )


# Search uploaded documents
@app.post("/search")
def search_documents(data: Question):

    results = search_chunks(data.question)

    if not results or not results["documents"][0]:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Upload a PDF first."
        )

    retrieved_documents = []

    for i, document in enumerate(results["documents"][0]):

        retrieved_documents.append({
            "content": document,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    return {
        "question": data.question,
        "results": retrieved_documents
    }
