
import os
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

from chunking import split_text
from vector_store import (
    store_chunks,
    search_chunks,
    list_documents,
    delete_document
)


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


# RAG question-answering endpoint
@app.post("/ask")
def ask_question(data: Question):

    # Step 1: Retrieve relevant chunks from ChromaDB
    results = search_chunks(data.question)

    if not results or not results["documents"][0]:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Upload a PDF first."
        )

    # Step 2: Prepare context and collect source pages
    context_parts = []
    source_pages = {}

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        source = metadata["source"]
        page = metadata.get("page")

        page_label = page if page is not None else "Unknown"

        context_parts.append(
            f"Source: {source}, Page: {page_label}\n{document}"
        )

        if source not in source_pages:
            source_pages[source] = set()

        if page is not None:
            source_pages[source].add(page)

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

    # Step 5: Format source references
    sources = []

    for filename, pages in source_pages.items():
        sources.append({
            "file": filename,
            "pages": sorted(pages)
        })

    # Step 6: Return answer and sources
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

        all_chunks = []
        all_page_numbers = []
        total_characters = 0

        # Extract and chunk each page separately
        for page_number, page in enumerate(pdf.pages, start=1):

            page_text = page.extract_text() or ""

            total_characters += len(page_text)

            if not page_text.strip():
                continue

            page_chunks = split_text(
                page_text,
                chunk_size=500,
                overlap=50
            )

            all_chunks.extend(page_chunks)

            # Associate every chunk with its original page
            all_page_numbers.extend(
                [page_number] * len(page_chunks)
            )

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF"
            )

        # Store chunks with page metadata
        stored_count = store_chunks(
            all_chunks,
            file.filename,
            all_page_numbers
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "total_characters": total_characters,
            "total_chunks": len(all_chunks),
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

        metadata = results["metadatas"][0][i]

        retrieved_documents.append({
            "content": document,
            "source": metadata["source"],
            "page": metadata.get("page"),
            "distance": results["distances"][0][i]
        })

    return {
        "question": data.question,
        "results": retrieved_documents
    }


# List uploaded documents
@app.get("/documents")
def get_documents():

    documents = list_documents()

    return {
        "total_documents": len(documents),
        "documents": documents
    }


# Delete an uploaded document
@app.delete("/documents/{filename}")
def remove_document(filename: str):

    deleted_count = delete_document(filename)

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "message": "Document deleted successfully",
        "filename": filename,
        "deleted_chunks": deleted_count
    }

