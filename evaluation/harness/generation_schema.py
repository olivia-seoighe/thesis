"""Typed records for end-to-end generation evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationQueryRecord:
    """A normalized golden query row for generation evaluation."""

    query_id: str
    query_text: str
    gold_answer: str
    category: str
    difficulty: int
    answerable: bool
    wording_type: str
    gold_services: tuple[str, ...] = ()
    gold_source_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResultRow:
    """One generation result row with answer, retrieval, and judge metrics."""

    run_id: str
    query_id: str
    query_text: str
    category: str
    difficulty: int
    answerable: bool
    mode: str
    decomposition_policy: str
    metadata_mode: str
    detected_services: str
    source_searched: str
    generated_answer: str
    gold_answer: str
    citations_count: int
    citation_sources: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    decomposition_used: bool
    decomposition_reason: str
    decomposition_branches: str
    branch_result_counts: str
    correctness_score: float
    completeness_score: float
    abstention_quality_score: float
    hallucination_flag: int
    hallucination_notes: str
    faithfulness_score: float
    citation_precision: float
    citation_recall_proxy: float
    gold_service_overlap: float
    gold_file_overlap: float
    judge_model: str
    judge_prompt_version: str
    judge_rationale: str
    score_confidence: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationAggregateRow:
    """Aggregated generation metrics for category/difficulty/overall sheets."""

    mode: str
    decomposition_policy: str
    n: int
    correctness_score: float
    completeness_score: float
    abstention_quality_score: float
    faithfulness_score: float
    citation_precision: float
    citation_recall_proxy: float
    gold_service_overlap: float
    gold_file_overlap: float
    hallucination_rate: float
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    error_rate: float
    group_value: str = ""
    group_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationFailureRow:
    """One failing or low-scoring generation result for debugging."""

    run_id: str
    query_id: str
    mode: str
    decomposition_policy: str
    correctness_score: float
    completeness_score: float
    faithfulness_score: float
    hallucination_flag: int
    error: str
    generated_answer: str
    gold_answer: str
    judge_rationale: str
    citations_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRunMeta:
    """Metadata persisted alongside a generation evaluation run."""

    run_id: str
    created_at_utc: str
    dataset_version: str
    dataset_hashes: dict[str, str]
    modes: tuple[str, ...]
    decomposition_policies: tuple[str, ...]
    generation_url: str
    query_count: int
    config: dict[str, Any]
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
