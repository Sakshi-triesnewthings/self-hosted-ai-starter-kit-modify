from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from config import settings
from models import HealthResponse, QueryRequest, QueryResponse
from rag_pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the pipeline once, on startup, and stash it on app.state.
    # Construction here (not at import time) means importing this module never
    # crashes the process, and the single instance is reused across requests.
    app.state.pipeline = RAGPipeline()
    yield
    # No explicit teardown: the Ollama/Qdrant clients hold no long-lived
    # connections that need closing.


app = FastAPI(
    title="Local RAG API",
    description="A FastAPI wrapper around a local Ollama + Qdrant RAG pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> RAGPipeline:
    """Dependency that hands endpoints the shared pipeline instance."""
    return request.app.state.pipeline


@app.get("/")
def root():
    return {
        "message": "RAG API is running",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "qdrant_collection": settings.qdrant_collection,
    }


@app.get("/health", response_model=HealthResponse)
def health(response: Response, pipeline: RAGPipeline = Depends(get_pipeline)):
    ollama_ok = pipeline.check_ollama()
    qdrant_ok = pipeline.check_qdrant()
    healthy = ollama_ok and qdrant_ok

    # Return 503 when degraded so orchestrators / uptime probes can detect it.
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama_reachable=ollama_ok,
        qdrant_reachable=qdrant_ok,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        return pipeline.query(request.question, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
