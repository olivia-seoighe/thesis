"""Shared predicate catalogs for graph traversal and ranking."""

from __future__ import annotations

from .types import QueryIntent

ALLOWED_PREDICATES_BY_INTENT: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.TOPOLOGY: (
        "OWNS_PROJECT",
        "TARGETS_FRAMEWORK",
        "REFERENCES_PACKAGE",
        "CONSUMES_TOPIC",
        "PRODUCES_TOPIC",
        "CALLS_API",
        "EXPOSES_API",
    ),
    QueryIntent.FLOW: (
        "OWNS_HANDLER",
        "OWNS_COMMAND",
        "OWNS_EVENT",
        "OWNS_SAGA",
        "HANDLES_COMMAND",
        "HANDLES_EVENT",
        "EMITS_EVENT",
        "SAGA_ORCHESTRATES_COMMAND",
        "SAGA_AWAITS_EVENT",
        "CONSUMES_TOPIC",
        "PRODUCES_TOPIC",
        "CALLS_API",
        "EXPOSES_API",
        "READS_TABLE",
        "WRITES_TABLE",
    ),
    QueryIntent.LOCAL_LOGIC: (
        "OWNS_HANDLER",
        "OWNS_COMMAND",
        "OWNS_EVENT",
        "OWNS_SAGA",
        "HANDLES_COMMAND",
        "HANDLES_EVENT",
        "ENFORCES_RULE",
        "USES_FEATURE_FLAG",
        "TRANSITIONS_STATUS",
    ),
    QueryIntent.CONFIG: (
        "OWNS_PROJECT",
        "TARGETS_FRAMEWORK",
        "REFERENCES_PACKAGE",
        "CONSUMES_TOPIC",
        "PRODUCES_TOPIC",
        "CALLS_API",
        "EXPOSES_API",
    ),
}

PREFERRED_PREDICATES_BY_INTENT: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.TOPOLOGY: ALLOWED_PREDICATES_BY_INTENT[QueryIntent.TOPOLOGY],
    QueryIntent.FLOW: (
        "HANDLES_COMMAND",
        "HANDLES_EVENT",
        "EMITS_EVENT",
        "SAGA_ORCHESTRATES_COMMAND",
        "SAGA_AWAITS_EVENT",
        "PRODUCES_TOPIC",
        "CONSUMES_TOPIC",
        "CALLS_API",
        "EXPOSES_API",
        "OWNS_HANDLER",
        "OWNS_COMMAND",
        "OWNS_EVENT",
        "OWNS_SAGA",
        "READS_TABLE",
        "WRITES_TABLE",
    ),
    QueryIntent.LOCAL_LOGIC: (
        "ENFORCES_RULE",
        "USES_FEATURE_FLAG",
        "TRANSITIONS_STATUS",
        "HANDLES_COMMAND",
        "HANDLES_EVENT",
        "OWNS_HANDLER",
        "OWNS_COMMAND",
        "OWNS_EVENT",
        "OWNS_SAGA",
    ),
    QueryIntent.CONFIG: ALLOWED_PREDICATES_BY_INTENT[QueryIntent.CONFIG],
}

PREDICATE_PRIORITY_BY_INTENT: dict[QueryIntent, dict[str, int]] = {
    QueryIntent.TOPOLOGY: {
        "TARGETS_FRAMEWORK": 0,
        "REFERENCES_PACKAGE": 0,
        "OWNS_PROJECT": 1,
        "CONSUMES_TOPIC": 0,
        "PRODUCES_TOPIC": 0,
        "CALLS_API": 1,
        "EXPOSES_API": 1,
        "OWNS_HANDLER": 2,
        "OWNS_COMMAND": 2,
        "OWNS_EVENT": 2,
        "OWNS_SAGA": 2,
    },
    QueryIntent.FLOW: {
        "HANDLES_COMMAND": 0,
        "HANDLES_EVENT": 0,
        "EMITS_EVENT": 0,
        "PRODUCES_TOPIC": 1,
        "CONSUMES_TOPIC": 1,
        "CALLS_API": 1,
        "EXPOSES_API": 1,
        "OWNS_HANDLER": 2,
        "OWNS_COMMAND": 2,
        "OWNS_EVENT": 2,
        "OWNS_SAGA": 2,
    },
}

RELATION_WEIGHTS_BY_INTENT: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.TOPOLOGY: {
        "TARGETS_FRAMEWORK": 1.24,
        "REFERENCES_PACKAGE": 1.18,
        "OWNS_PROJECT": 1.1,
        "CONSUMES_TOPIC": 1.25,
        "PRODUCES_TOPIC": 1.22,
        "CALLS_API": 1.12,
        "EXPOSES_API": 1.1,
        "OWNS_HANDLER": 1.05,
        "OWNS_COMMAND": 1.05,
        "OWNS_EVENT": 1.05,
        "OWNS_SAGA": 1.05,
    },
    QueryIntent.FLOW: {
        "HANDLES_COMMAND": 1.18,
        "HANDLES_EVENT": 1.16,
        "EMITS_EVENT": 1.14,
        "SAGA_ORCHESTRATES_COMMAND": 1.12,
        "SAGA_AWAITS_EVENT": 1.10,
        "PRODUCES_TOPIC": 1.1,
        "CONSUMES_TOPIC": 1.1,
        "CALLS_API": 1.1,
        "EXPOSES_API": 1.06,
        "TRANSITIONS_STATUS": 1.08,
    },
    QueryIntent.LOCAL_LOGIC: {
        "ENFORCES_RULE": 1.2,
        "USES_FEATURE_FLAG": 1.12,
        "READS_TABLE": 1.08,
        "WRITES_TABLE": 1.08,
        "TRANSITIONS_STATUS": 1.1,
    },
}


def allowed_predicates(intent: QueryIntent) -> list[str]:
    return list(ALLOWED_PREDICATES_BY_INTENT.get(intent, ()))


def preferred_predicates(intent: QueryIntent) -> list[str]:
    return list(PREFERRED_PREDICATES_BY_INTENT.get(intent, ()))


def predicate_priority(intent: QueryIntent) -> dict[str, int]:
    return dict(PREDICATE_PRIORITY_BY_INTENT.get(intent, {}))


def relation_weights(intent: QueryIntent) -> dict[str, float]:
    return dict(RELATION_WEIGHTS_BY_INTENT.get(intent, {}))
