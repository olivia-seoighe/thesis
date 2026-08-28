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
        "many",
        "each",
        "service",
        "services",
        "implement",
        "implements",
        "declare",
        "declares",
        "declared",
    }
)
