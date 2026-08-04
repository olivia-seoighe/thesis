from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    top_k: int = Field(5, description="Number of top results to return")
    sources: List[str] = Field(
        default_factory=list,
        description="Source filters. Empty list = search all sources.",
    )
    match_all: bool = Field(
        False,
        description="If true, all query terms must exist somewhere in the document "
        "(across any of its chunks). Default is false (OR logic).",
    )
    max_chunks_per_document: int | None = Field(
        3,
        description="Maximum chunks per document for diversity. Default 3. None = no limit.",
    )
    entity_filter: str | None = Field(
        None,
        description="Optional entity filter for vector search. Matches document title and content.",
    )



class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    document_id: str = Field(..., description="ID of the source document")
    document_title: str = Field(..., description="Title of the source document")
    last_modified_date: str = Field(..., description="Last modified date of the source document")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(..., description="Document source")
    url: str = Field("", description="URL of the source file")
    source_code: str = Field("", description="Original source code of the file")


class SearchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    chunks: List[RetrievedChunk] = Field(default_factory=list)
    total_results: int = Field(...)
    search_duration_ms: float = Field(...)
    embedding_duration_ms: float = Field(...)
    model_used: str = Field(...)
    source_searched: str = Field(...)