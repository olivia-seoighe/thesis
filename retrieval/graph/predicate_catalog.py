"""Shared predicate catalogs for graph traversal and ranking."""

from __future__ import annotations

from .types import QueryIntent

LOCAL_PREDICATES: tuple[str, ...] = (
    "OWNS_HANDLER",
    "OWNS_COMMAND",
    "OWNS_EVENT",
    "OWNS_SAGA",
    "HANDLES_COMMAND",
    "HANDLES_EVENT",
    "EMITS_EVENT",
    "SAGA_ORCHESTRATES_COMMAND",
    "SAGA_AWAITS_EVENT",
    "READS_TABLE",
    "WRITES_TABLE",
    "ENFORCES_RULE",
    "USES_FEATURE_FLAG",
    "TRANSITIONS_STATUS",
)

GLOBAL_PREDICATES: tuple[str, ...] = (
    "CONTAINS_PACKAGE",
    "TARGETS_FRAMEWORK",
    "CONSUMES_TOPIC",
    "PRODUCES_TOPIC",
    "CALLS_API",
    "EXPOSES_API",
)

ALLOWED_PREDICATES_BY_INTENT: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.TOPOLOGY: GLOBAL_PREDICATES,
    QueryIntent.LOCAL_LOGIC: LOCAL_PREDICATES,
}

def allowed_predicates(intent: QueryIntent) -> list[str]:
    return list(ALLOWED_PREDICATES_BY_INTENT.get(intent, ()))
