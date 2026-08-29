from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None
    top_k: int = 5
    mode: str = "hybrid"              # retrieval mode: hybrid | vector | keyword | graph
    model: Optional[str] = None       # overrides OPENAI_MODEL
    conversation_id: Optional[str] = None  # pass to continue a conversation


class Citation(BaseModel):
    title: str
    url: str
    score: float
    chunk_text: str
    source_code: str = ""
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    citations: List[Citation]
    conversation_id: str
    model_used: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    decomposition_used: bool = False
    decomposition_reason: str = ""
    decomposition_branches: List[str] = Field(default_factory=list)
    branch_result_counts: Dict[str, int] = Field(default_factory=dict)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: str


class Conversation(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: str


class EmbeddingPoint(BaseModel):
    """A single point in the 2-D embedding visualisation."""
    id: str
    label: str
    x: float
    y: float
    score: float
    type: str       # "query" | "chunk"
    source: str = ""


class VizResponse(BaseModel):
    points: List[EmbeddingPoint]
    note: str = ""
