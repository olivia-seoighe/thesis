"""Query intent + mention extraction for graph retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable

from retrieval.query_processor import expand_query_text, normalize_query_token

from .lexicon import (
    COMMON_QUERY_STOPWORDS,
)
from .types import REPO_LABEL, SEEDABLE_NODE_LABELS, EntityMention, QueryIntent, SeedRequest

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAX_QUERY_TOKEN_MENTIONS = 12
_LABEL_HINTS_BY_TOKEN: dict[str, tuple[str, ...]]
_LOCAL_LABELS: frozenset[str] = frozenset(
    {
        "HANDLER",
        "COMMAND",
        "EVENT",
        "SAGA",
        "FEATURE_FLAG",
        "BUSINESS_RULE",
        "STATUS_CODE",
        "TABLE",
    }
)
_GLOBAL_LABELS: frozenset[str] = frozenset(
    {
        REPO_LABEL,
        "KAFKA_TOPIC",
        "API",
        "NUGET_PACKAGE",
        "FRAMEWORK",
    }
)


@dataclass(frozen=True)
class _AliasEntry:
    source: str
    short_forms: tuple[str, ...]
    long_forms: tuple[str, ...]


def build_query_seed_mentions(
    query: str,
    service_catalogue: Iterable[tuple[str, Iterable[str]]],
) -> tuple[str, list[EntityMention]]:
    normalized_query = normalize_query_text(query)
    preprocessed_query = expand_service_aliases_in_query(normalized_query, service_catalogue)
    mentions = extract_mentions(preprocessed_query)
    expanded_mentions = expand_aliases(mentions, service_catalogue, preprocessed_query)
    return preprocessed_query, expanded_mentions


def normalize_query_text(query: str) -> str:
    return " ".join(query.split())


def expand_service_aliases_in_query(
    query: str,
    service_catalogue: Iterable[tuple[str, Iterable[str]]],
) -> str:
    entries = tuple(
        _AliasEntry(
            source=canonical_service,
            short_forms=tuple(aliases),
            long_forms=(),
        )
        for canonical_service, aliases in service_catalogue
        if canonical_service
    )
    if not entries:
        return query
    return expand_query_text(query, entries)


def extract_mentions(query: str) -> list[EntityMention]:
    return _dedupe_mentions(_extract_query_token_mentions(query))


def expand_aliases(
    mentions: list[EntityMention],
    service_catalogue: Iterable[tuple[str, Iterable[str]]],
    query_text: str | None = None,
) -> list[EntityMention]:
    alias_map, aliases_by_service = _build_service_alias_maps(service_catalogue)
    expanded: list[EntityMention] = []
    for mention in mentions:
        canonical_service = alias_map.get(mention.normalized)
        if canonical_service:
            expanded.append(_repo_mention(canonical_service, max(mention.confidence, 0.95)))
        expanded.append(mention)
    expanded.extend(_extract_query_service_mentions(query_text=query_text, aliases_by_service=aliases_by_service))
    return _dedupe_mentions(expanded)


def build_seed_candidates(
    query: str,
    mentions: list[EntityMention],
    *,
    source_filters: list[str] | None = None,
) -> SeedRequest:
    deduped_mentions = tuple(_dedupe_mentions(mentions))
    return SeedRequest(
        query=query,
        intent=infer_intent_from_mentions(deduped_mentions),
        mentions=deduped_mentions,
        source_filters=tuple(source_filters or ()),
    )


def infer_intent_from_mentions(mentions: tuple[EntityMention, ...]) -> QueryIntent:
    labels = {mention.preferred_label for mention in mentions if mention.preferred_label}
    return infer_intent_from_seed_labels(labels, fallback_intent=QueryIntent.GENERAL)


def infer_intent_from_seed_labels(
    labels: set[str] | frozenset[str],
    *,
    fallback_intent: QueryIntent = QueryIntent.GENERAL,
) -> QueryIntent:
    normalized_labels = {label.upper() for label in labels if label}
    has_local_labels = bool(normalized_labels & _LOCAL_LABELS)
    has_global_labels = bool(normalized_labels & _GLOBAL_LABELS)

    if has_local_labels and has_global_labels:
        if fallback_intent in {QueryIntent.LOCAL_LOGIC, QueryIntent.TOPOLOGY}:
            return fallback_intent
        return QueryIntent.TOPOLOGY
    if has_local_labels and not has_global_labels:
        return QueryIntent.LOCAL_LOGIC
    if has_global_labels:
        return QueryIntent.TOPOLOGY
    return fallback_intent


def _extract_query_token_mentions(query: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    seen: set[str] = set()

    for token in _QUERY_TOKEN_RE.findall(query):
        for normalized in _normalized_query_token_forms(token):
            if not normalized or normalized in seen:
                continue
            if normalized in COMMON_QUERY_STOPWORDS:
                continue
            if len(normalized) < 3 or normalized.isdigit():
                continue
            seen.add(normalized)
            label_hints = _LABEL_HINTS_BY_TOKEN.get(normalized, ())
            if not label_hints:
                mentions.append(_mention(normalized, preferred_label=None, confidence=0.58))
            for label in label_hints:
                mentions.append(_mention(normalized, preferred_label=label, confidence=0.86))
            if len(seen) >= _MAX_QUERY_TOKEN_MENTIONS:
                return mentions
    return mentions


def _normalized_query_token_forms(token: str) -> tuple[str, ...]:
    normalized = normalize_query_token(token)
    if not normalized:
        return ()
    singular = ""
    if normalized.endswith("s") and not normalized.endswith("ss"):
        candidate = normalized[:-1]
        if candidate in _LABEL_HINTS_BY_TOKEN:
            singular = candidate
    if singular and singular != normalized:
        return (normalized, singular)
    return (normalized,)


def _build_label_hints_by_token(seedable_labels: set[str] | frozenset[str]) -> dict[str, tuple[str, ...]]:
    hints: dict[str, set[str]] = {}
    for label in seedable_labels:
        normalized = normalize_query_token(label)
        if not normalized:
            continue
        for token in normalized.split("-"):
            if len(token) >= 3:
                hints.setdefault(token, set()).add(label)
    return {token: tuple(sorted(labels)) for token, labels in hints.items()}


_LABEL_HINTS_BY_TOKEN = _build_label_hints_by_token(SEEDABLE_NODE_LABELS)


def _mention(text: str, *, preferred_label: str | None, confidence: float) -> EntityMention:
    normalized = normalize_query_token(text)
    return EntityMention(
        text=text.strip(),
        normalized=normalized,
        preferred_label=preferred_label,
        confidence=confidence,
    )


def _dedupe_mentions(mentions: list[EntityMention]) -> list[EntityMention]:
    best_by_key: dict[tuple[str, str | None], EntityMention] = {}
    for mention in mentions:
        if not mention.normalized:
            continue
        key = (mention.normalized, mention.preferred_label)
        previous = best_by_key.get(key)
        if previous is None or mention.confidence > previous.confidence:
            best_by_key[key] = mention
    return sorted(
        best_by_key.values(),
        key=lambda item: (-item.confidence, item.preferred_label or "", item.normalized),
    )


def _build_service_alias_maps(
    service_catalogue: Iterable[tuple[str, Iterable[str]]],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    alias_map: dict[str, str] = {}
    aliases_by_service: dict[str, set[str]] = {}
    for canonical_service, aliases in service_catalogue:
        canonical = normalize_query_token(canonical_service)
        if canonical:
            alias_map[canonical] = canonical_service
            aliases_by_service.setdefault(canonical_service, set()).add(canonical)
        for alias in aliases:
            normalized = normalize_query_token(alias)
            if normalized:
                alias_map[normalized] = canonical_service
                aliases_by_service.setdefault(canonical_service, set()).add(normalized)
    return alias_map, aliases_by_service


def _extract_query_service_mentions(
    *,
    query_text: str | None,
    aliases_by_service: dict[str, set[str]],
) -> list[EntityMention]:
    normalized_query = normalize_query_token(query_text or "")
    if not normalized_query:
        return []

    mentions: list[EntityMention] = []
    for canonical_service, alias_tokens in aliases_by_service.items():
        if any(token and token in normalized_query for token in alias_tokens):
            mentions.append(_repo_mention(canonical_service, 0.96))
    return mentions


def _repo_mention(service_name: str, confidence: float) -> EntityMention:
    return EntityMention(
        text=service_name,
        normalized=normalize_query_token(service_name),
        preferred_label=REPO_LABEL,
        confidence=confidence,
    )
