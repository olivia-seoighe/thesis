"""Shared lexical filters for graph query parsing."""

from __future__ import annotations

COMMON_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "what",
        "which",
        "with",
    }
)

FLOW_INTENT_TERMS: tuple[str, ...] = ("flow", "path", "emit", "consume", "publish", "call")
TOPOLOGY_INTENT_TERMS: tuple[str, ...] = ("which services", "across services", "depends on", "dependency")
CONFIG_INTENT_TERMS: tuple[str, ...] = ("config", "setting", "appsettings")
LOCAL_INTENT_TERMS: tuple[str, ...] = ("inside", "within", "handler", "command", "event")
