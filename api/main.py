from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import logging
import sys
import os
import asyncio
from urllib.parse import urlsplit
from dotenv import load_dotenv

# Load environment variables from the project root first.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Add parent directory to path to import rag module
sys.path.append(PROJECT_ROOT)

from rag.retrieval_pipeline import retrieve_and_answer
from rag.chroma_connection import create_chroma_client
from api.execution_limits import (
    CapacityExceededError,
    RagExecutionTimeoutError,
    get_rag_execution_limiter,
    positive_number,
)

logger = logging.getLogger("constitution_gpt.api")


def get_frontend_origins() -> list[str]:
    """Read and validate the comma-separated browser origins allowed by CORS."""
    configured_origins = os.getenv("FRONTEND_ORIGINS", "")
    origins = []

    for configured_origin in configured_origins.split(","):
        origin = configured_origin.strip().rstrip("/")
        if not origin:
            continue

        parsed = urlsplit(origin)
        is_origin = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )
        if not is_origin:
            raise RuntimeError(
                "Invalid FRONTEND_ORIGINS entry. Use comma-separated origins "
                "such as https://example.com (without paths)."
            )

        if origin not in origins:
            origins.append(origin)

    if not origins:
        raise RuntimeError(
            "FRONTEND_ORIGINS is required. For local development, set it to "
            "http://localhost:3000,http://127.0.0.1:3000."
        )

    return origins

app = FastAPI(
    title="Constitution GPT API",
    description="AI-Powered Constitutional Intelligence API for Nepal's Constitution",
    version="1.0.0"
)

# Only browser origins explicitly configured for this environment may call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    
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
            "/health/live": "Liveness check",
            "/health/ready": "Dependency readiness check",
            "/api/chat": "Query the Constitution (POST)",
            "/docs": "Interactive API documentation",
        }
    }


@app.get("/health/live")
async def liveness():
    """Confirm that the API process can serve requests."""
    return {
        "status": "alive",
        "service": "Constitution GPT API"
    }


@app.get("/health")
@app.get("/health/ready")
async def readiness():
    """Confirm that dependencies required by chat requests are available."""
    try:
        await asyncio.wait_for(
            run_in_threadpool(lambda: create_chroma_client().heartbeat()),
            timeout=positive_number("CHROMA_HEALTH_TIMEOUT_SECONDS", 5),
        )
    except Exception:
        logger.exception("Readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "service": "Constitution GPT API"},
        )

    return {
        "status": "ready",
        "service": "Constitution GPT API",
        "dependencies": {"chroma": "ok"},
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
        # The RAG stack uses synchronous SDK clients. Running it in FastAPI's
        # worker pool prevents one slow model/database call from blocking the
        # event loop for every concurrent request.
        try:
            answer = await get_rag_execution_limiter().run(
                retrieve_and_answer,
                request.question,
                False,
            )
        except CapacityExceededError:
            raise HTTPException(
                status_code=429,
                detail="The service is at capacity. Please try again shortly.",
                headers={"Retry-After": "2"},
            )
        except RagExecutionTimeoutError:
            raise HTTPException(
                status_code=504,
                detail="The request exceeded its processing deadline.",
            )
        
        return QueryResponse(
            question=request.question,
            answer=answer
        )
    
    except HTTPException:
        raise
    except Exception:
        # Never return provider errors, stack details, credentials, prompts, or
        # internal topology to an untrusted client.
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question at this time."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
