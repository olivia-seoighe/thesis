"""Shared retrieval corpus normalization."""

from __future__ import annotations

VALID_RETRIEVAL_CORPORA: tuple[str, ...] = ("summaries", "code", "all")
VALID_RETRIEVAL_CORPORA: tuple[str, ...] = ("summaries", "code", "all")


# Normalizes corpus aliases to the names stored in the retrieval index.
def normalize_retrieval_corpus(value: str | None, *, default: str = "summaries") -> str:
    token = (value or default).strip().lower()
    if token in {"summary", "summaries"}:
        return "summaries"
    if token in {"source_code", "code"}:
        return "code"
    if token == "all":
        return "all"
    raise ValueError(f"retrieval corpus must be one of: {', '.join(VALID_RETRIEVAL_CORPORA)}")
