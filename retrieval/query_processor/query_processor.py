"""Reusable query normalization and expansion for service-aware routing."""

from __future__ import annotations

import re
from typing import Protocol


class ServiceAliasEntry(Protocol):
    source: str
    short_forms: tuple[str, ...]
    long_forms: tuple[str, ...]


def normalize_query_token(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[\s_]+", "-", lowered)
    lowered = re.sub(r"[^a-z0-9-]", "", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-")


def phrase_present(query_text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", query_text) is not None


def expand_query_text(query_text: str, entries: tuple[ServiceAliasEntry, ...]) -> str:
    query_normalized = normalize_query_token(query_text)
    query_lower = query_text.lower()
    additions: set[str] = set()

    for entry in entries:
        matched_short = any(
            phrase_present(query_normalized, normalize_query_token(value))
            or phrase_present(query_lower, value.lower())
            for value in entry.short_forms
        )
        matched_long = any(
            phrase_present(query_normalized, normalize_query_token(value))
            or phrase_present(query_lower, value.lower())
            for value in entry.long_forms
        )

        if matched_short:
            additions.update(entry.long_forms)
            if entry.source:
                additions.add(entry.source)
        if matched_long:
            additions.update(entry.short_forms)
            if entry.source:
                additions.add(entry.source)

    if not additions:
        return query_text
    return f"{query_text} {' '.join(sorted(additions))}"
