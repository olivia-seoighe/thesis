from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str
    url: str
    source_code: str = ""
    last_modified_date: Optional[str] = None
    source_refs: str = ""
    retrieval_corpus: str = "summaries"


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    document: Document
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedDocumentChunk:
    chunk: DocumentChunk
    embedding: List[float]
    embedding_model: str
