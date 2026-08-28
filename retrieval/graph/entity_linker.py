"""Query intent + mention extraction for graph retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable

from retrieval.query_processor import normalize_query_token

from .lexicon import (
    COMMON_QUERY_STOPWORDS,
    CONFIG_INTENT_TERMS,
    FLOW_INTENT_TERMS,
    LOCAL_INTENT_TERMS,
    TOPOLOGY_INTENT_TERMS,
)
from .types import (
    API_LABEL,
    COMMAND_LABEL,
    EVENT_LABEL,
    HANDLER_LABEL,
    REPO_LABEL,
    SAGA_LABEL,
    TOPIC_LABEL,
    EntityMention,
    QueryIntent,
    SeedRequest,
)

_SERVICE_RE = re.compile(r"\b([a-z0-9][a-z0-9-]*-service)\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r'["\']([^"\']+)["\']')
_TOPIC_HINT_RE = re.compile(r"\btopics?\s+([a-z0-9._-]+)\b", re.IGNORECASE)
_API_HINT_RE = re.compile(r"\bapi\s+([/\w{}-]+)\b", re.IGNORECASE)
_REPO_TOKEN_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)+)\b", re.IGNORECASE)
_REPO_SUFFIXES = ("-service", "-api", "-system", "-bff", "-workflow")
_SYMBOL_HINT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{3,}(?:Handler|Command|Event|Saga|Request))\b")
_SYMBOL_SUFFIX_TO_LABEL: tuple[tuple[str, str], ...] = (
    ("Handler", HANDLER_LABEL),
    ("Saga", SAGA_LABEL),
    ("Event", EVENT_LABEL),
    ("Command", COMMAND_LABEL),
    ("Request", COMMAND_LABEL),
)


def detect_intent(query: str) -> QueryIntent:
    lowered = query.lower()
    if any(term in lowered for term in CONFIG_INTENT_TERMS):
        return QueryIntent.CONFIG
    if any(term in lowered for term in TOPOLOGY_INTENT_TERMS):
        return QueryIntent.TOPOLOGY
    if any(term in lowered for term in FLOW_INTENT_TERMS):
        return QueryIntent.FLOW
    if any(term in lowered for term in LOCAL_INTENT_TERMS):
        return QueryIntent.LOCAL_LOGIC
    return QueryIntent.GENERAL


def extract_mentions(query: str) -> list[EntityMention]:
    mentions = (
        _extract_service_mentions(query)
        + _extract_repo_token_mentions(query)
        + _extract_symbol_mentions(query)
        + _extract_topic_mentions(query)
        + _extract_api_mentions(query)
        + _extract_quoted_mentions(query)
    )
    return _dedupe_mentions(mentions)


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
    return SeedRequest(
        query=query,
        intent=detect_intent(query),
        mentions=tuple(_dedupe_mentions(mentions)),
        source_filters=tuple(source_filters or ()),
    )


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


def _extract_service_mentions(query: str) -> list[EntityMention]:
    return [_mention(match, preferred_label=REPO_LABEL, confidence=0.98) for match in _SERVICE_RE.findall(query)]


def _extract_repo_token_mentions(query: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    for token in _REPO_TOKEN_RE.findall(query):
        if token.lower().endswith(_REPO_SUFFIXES):
            mentions.append(_mention(token, preferred_label=REPO_LABEL, confidence=0.92))
    return mentions


def _extract_symbol_mentions(query: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    for token in _SYMBOL_HINT_RE.findall(query):
        label = COMMAND_LABEL
        for suffix, mapped_label in _SYMBOL_SUFFIX_TO_LABEL:
            if token.endswith(suffix):
                label = mapped_label
                break
        mentions.append(_mention(token, preferred_label=label, confidence=0.94))
    return mentions


def _extract_topic_mentions(query: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    for match in _TOPIC_HINT_RE.findall(query):
        normalized_match = normalize_query_token(match)
        if not normalized_match or normalized_match in COMMON_QUERY_STOPWORDS:
            continue
        mentions.append(_mention(match, preferred_label=TOPIC_LABEL, confidence=0.92))
    return mentions


def _extract_api_mentions(query: str) -> list[EntityMention]:
    return [_mention(match, preferred_label=API_LABEL, confidence=0.9) for match in _API_HINT_RE.findall(query)]


def _extract_quoted_mentions(query: str) -> list[EntityMention]:
    return [_mention(match, preferred_label=None, confidence=0.8) for match in _QUOTED_RE.findall(query)]


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
