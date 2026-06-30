from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class Source(BaseModel):
    text: str = Field(..., description="The chunk of text retrieved from the document")
    score: float = Field(..., description="Similarity score from the vector search")
    metadata: dict = Field(default_factory=dict, description="Any metadata attached to the chunk")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    qdrant_reachable: bool