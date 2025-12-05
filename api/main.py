from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add parent directory to path to import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieval_pipeline import retrieve_and_answer

app = FastAPI(
    title="Constitution GPT API",
    description="AI-Powered Constitutional Intelligence API for Nepal's Constitution",
    version="1.0.0"
)

# Configure CORS to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How is the Prime Minister elected in Nepal?"
            }
        }


class QueryResponse(BaseModel):
    question: str
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How is the Prime Minister elected in Nepal?",
                "answer": "📘 Part 7 – Federal Executive\nArticle 76 – Constitution of Council of Ministers..."
            }
        }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Constitution GPT API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/api/chat": "Query the Constitution (POST)",
            "/docs": "Interactive API documentation",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Constitution GPT API"
    }


@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Query the Constitution of Nepal using RAG.
    
    - **question**: Your question about the Constitution of Nepal
    
    Returns a structured answer with proper citations and hierarchical structure.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Call the RAG pipeline (verbose=False to avoid console output)
        answer = retrieve_and_answer(request.question, verbose=False)
        
        return QueryResponse(
            question=request.question,
            answer=answer
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
