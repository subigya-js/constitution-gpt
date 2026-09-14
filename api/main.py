from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import logging
import sys
import os
import asyncio
from urllib.parse import urlsplit
from uuid import UUID
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from the project root first.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Add parent directory to path to import rag module
sys.path.append(PROJECT_ROOT)

from rag.research_assistant import ResearchMode, answer_research_question
from rag.chroma_connection import create_chroma_client
from api.chat_repository import ChatRepository
from api.execution_limits import (
    CapacityExceededError,
    RagExecutionTimeoutError,
    get_rag_execution_limiter,
    positive_number,
)

logger = logging.getLogger("constitution_gpt.api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize server-side dependencies once per API process."""
    chat_repository = ChatRepository()
    await chat_repository.start()
    application.state.chat_repository = chat_repository
    try:
        yield
    finally:
        await chat_repository.close()


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
    description="Source-grounded constitutional and Nepalese legal research API",
    version="1.1.0",
    lifespan=lifespan,
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
    conversation_id: UUID | None = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How is the Prime Minister elected in Nepal?",
                "conversation_id": "f8d53b9b-dbae-4477-b7a4-bf7c30c2b411",
            }
        }


class SourceResponse(BaseModel):
    title: str
    url: str
    source_type: Literal["web"]


class QueryResponse(BaseModel):
    question: str
    answer: str
    conversation_id: UUID
    resolved_question: str
    mode: ResearchMode
    sources: list[SourceResponse]
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How is the Prime Minister elected in Nepal?",
                "answer": "📘 Part 7 – Federal Executive\nArticle 76 – Constitution of Council of Ministers...",
                "conversation_id": "f8d53b9b-dbae-4477-b7a4-bf7c30c2b411",
                "resolved_question": "How is the Prime Minister elected in Nepal?",
                "mode": "constitutional",
                "sources": [],
            }
        }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Constitution GPT API",
        "version": "1.1.0",
        "endpoints": {
            "/": "API information",
            "/health/live": "Liveness check",
            "/health/ready": "Dependency readiness check",
            "/api/chat": "Ask a constitutional or Nepalese legal research question (POST)",
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
async def readiness(request: Request):
    """Confirm that dependencies required by chat requests are available."""
    try:
        await asyncio.gather(
            asyncio.wait_for(
                run_in_threadpool(lambda: create_chroma_client().heartbeat()),
                timeout=positive_number("CHROMA_HEALTH_TIMEOUT_SECONDS", 5),
            ),
            asyncio.wait_for(
                request.app.state.chat_repository.check_connection(),
                timeout=positive_number("DB_HEALTH_TIMEOUT_SECONDS", 5),
            ),
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
        "dependencies": {"chroma": "ok", "postgres": "ok"},
    }


@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest, http_request: Request):
    """
    Ask a conversational or source-grounded Nepalese legal research question.
    
    - **question**: Your question about the Constitution of Nepal
    
    Returns the answer, conversation identity, research mode, and web sources.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        conversation_id = await http_request.app.state.chat_repository.ensure_conversation(
            request.conversation_id
        )
        history = await http_request.app.state.chat_repository.get_recent_messages(
            conversation_id
        )

        # The research stack uses synchronous SDK clients. Running it in FastAPI's
        # worker pool prevents one slow model/database call from blocking the
        # event loop for every concurrent request.
        try:
            result = await get_rag_execution_limiter().run(
                answer_research_question,
                request.question,
                history,
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
        
        serialized_sources = [source.model_dump() for source in result.sources]
        await http_request.app.state.chat_repository.save_interaction(
            request.question,
            result.answer,
            conversation_id=conversation_id,
            resolved_question=result.resolved_question,
            mode=result.mode,
            sources=serialized_sources,
        )

        return QueryResponse(
            question=request.question,
            answer=result.answer,
            conversation_id=conversation_id,
            resolved_question=result.resolved_question,
            mode=result.mode,
            sources=[SourceResponse(**source) for source in serialized_sources],
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
