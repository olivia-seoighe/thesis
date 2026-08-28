"""Typed records used across dataset building and retrieval evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QueryRecord:
    """A normalized query row for retrieval evaluation."""

    query_id: str
    query_text: str
    category: str
    difficulty: int
    wording_type: str
    services: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QrelRecord:
    """A document-level relevance judgment for one query."""

    query_id: str
    doc_id: str
    service: str
    file_path: str
    relevance: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    """A raw retrieval hit from a baseline strategy call."""

    query_id: str
    category: str
    strategy: str
    rank: int
    doc_id: str
    chunk_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalErrorRow:
    """A retrieval request error captured during evaluation."""

    run_id: str
    query_id: str
    category: str
    strategy: str
    k: int
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalDecisionRow:
    """Per-query metadata-routing decision for service-aware retrieval."""

    run_id: str
    query_id: str
    category: str
    difficulty: int
    strategy: str
    query: str
    detected_services: tuple[str, ...]
    metadata_mode: str
    filter_services: tuple[str, ...]
    boost_services: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResultRow:
    """One strategy result row with retrieval metrics and routing metadata."""

    run_id: str
    query_id: str
    query_text: str
    category: str
    difficulty: int
    strategy: str
    k: int
    metadata_mode: str
    detected_services: str
    recall: float
    precision: float
    f1: float
    mrr: float
    ndcg: float
    relevant_count: int
    retrieved_count: int
    hit_count: int
    error: str
    graph_escalation_count: int = 0
    graph_hops_executed: int = 0
    graph_nodes_visited: int = 0
    graph_paths_examined: int = 0
    graph_stop_reason: str = ""
    graph_total_latency_ms: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricRow:
    """A single metric value computed per query and strategy."""

    run_id: str
    query_id: str
    category: str
    strategy: str
    k: int
    metric: str
    value: float
    relevant_count: int
    retrieved_count: int
    hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMeta:
    """Metadata persisted alongside a retrieval evaluation run."""

    run_id: str
    created_at_utc: str
    dataset_version: str
    dataset_hashes: dict[str, str]
    strategies: tuple[str, ...]
    k_values: tuple[int, ...]
    retrieval_url: str
    query_count: int
    qrel_count: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
