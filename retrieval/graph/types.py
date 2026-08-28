"""Shared graph retrieval data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

REPO_LABEL = "REPO"
FRAMEWORK_LABEL = "FRAMEWORK"
NUGET_PACKAGE_LABEL = "NUGET_PACKAGE"
HANDLER_LABEL = "HANDLER"
COMMAND_LABEL = "COMMAND"
EVENT_LABEL = "EVENT"
SAGA_LABEL = "SAGA"
TOPIC_LABEL = "KAFKA_TOPIC"
API_LABEL = "API"

SEEDABLE_NODE_LABELS: frozenset[str] = frozenset(
    {
        TOPIC_LABEL,
        API_LABEL,
        REPO_LABEL,
        FRAMEWORK_LABEL,
        NUGET_PACKAGE_LABEL,
        HANDLER_LABEL,
        COMMAND_LABEL,
        EVENT_LABEL,
        SAGA_LABEL,
    }
)


class QueryIntent(StrEnum):
    TOPOLOGY = "TOPOLOGY"
    LOCAL_LOGIC = "LOCAL_LOGIC"
    GENERAL = "GENERAL"


class TopologyScope(StrEnum):
    GLOBAL = "global"
    SERVICE_SCOPED = "service-scoped"
    TARGETED_MULTI_SERVICE = "targeted-multi-service"


@dataclass(frozen=True)
class EntityMention:
    text: str
    normalized: str
    preferred_label: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class SeedRequest:
    query: str
    intent: QueryIntent
    mentions: tuple[EntityMention, ...]
    source_filters: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphNodeRef:
    node_key: str
    node_label: str
    node_name: str
    confidence: float
    evidence_count: int


@dataclass(frozen=True)
class SeedMatch:
    mention: EntityMention
    node: GraphNodeRef
    match_score: float
    match_reason: str


@dataclass(frozen=True)
class SeedResolution:
    intent: QueryIntent
    seed_request: SeedRequest
    matches: tuple[SeedMatch, ...]
    unresolved_mentions: tuple[EntityMention, ...]


@dataclass(frozen=True)
class QueryFilters:
    source_filters: tuple[str, ...] = ()
    include_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraversalBudget:
    max_hops: int
    max_nodes_per_hop: int
    max_edges_per_node: int
    global_path_budget: int
    global_node_budget: int
    max_latency_ms: int


@dataclass(frozen=True)
class TraversalPlan:
    intent: QueryIntent
    seed_node_keys: tuple[str, ...]
    budget: TraversalBudget
    target_results: int = 5


@dataclass(frozen=True)
class TraversalState:
    intent: QueryIntent
    current_hop: int
    budget: TraversalBudget
    escalations: int
    visited_nodes: int
    visited_paths: int
    target_results: int = 5
    elapsed_ms: float = 0.0
    avg_confidence: float = 0.0
    distinct_source_kinds: int = 0
    ast_path_count: int = 0
    newly_added_paths: int = 0
    top_hop_frontier_size: int = 0
    stop_reason: str = "continue"


@dataclass(frozen=True)
class TraversalEscalationStep:
    from_budget: dict[str, int]
    to_budget: dict[str, int]
    reason: str
    at_hop: int
    elapsed_ms: float


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str


@dataclass(frozen=True)
class EvidenceBundle:
    evidence_count: int
    source_kinds: tuple[str, ...]
    has_line_bounds: bool
    tiers: tuple[str, ...]


@dataclass(frozen=True)
class GraphPath:
    start_node_key: str
    end_node_key: str
    predicates: tuple[str, ...]
    hops: int
    confidence: float
    evidence: EvidenceBundle


@dataclass(frozen=True)
class NodeCandidate:
    node: GraphNodeRef
    document_id: str
    chunk_id: str
    source: str
    document_title: str
    text: str
    source_code: str
    metadata: dict[str, Any]
    url: str
    last_modified_date: str
    evidence: EvidenceBundle
    path_score: float


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    source_code: str
    source: str
    url: str
    last_modified_date: str
    metadata: dict[str, Any]
    score: float
    score_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingContext:
    intent: QueryIntent
    decay_lambda: float = 0.35
    relation_weights: dict[str, float] = field(default_factory=dict)
    query_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CypherSpec:
    query: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class GraphTraversalMeta:
    intent: str
    hop_policy_mode: str
    initial_budget: dict[str, int]
    final_budget: dict[str, int]
    escalation_count: int
    escalation_steps: list[dict[str, Any]]
    escalation_reason: str
    hops_executed: int
    frontier_sizes_by_hop: dict[str, int]
    nodes_visited: int
    paths_examined: int
    stop_reason: str
    timing_ms: dict[str, float]


class GraphReader(Protocol):
    async def fetch(self, query: str, *args: Any) -> list[Any]:
        ...
