"""Service-aware retrieval planning from query text and service catalogue."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from retrieval.query_processor import expand_query_text, normalize_query_token, phrase_present
from retrieval.strategies.rrf import reciprocal_rank

SERVICE_AWARE_SUFFIX = "-service-aware"


class MetadataMode(StrEnum):
    GLOBAL = "GLOBAL"
    HARD_FILTER = "HARD_FILTER"
    BOOST = "BOOST"


@dataclass(frozen=True)
class ServiceResolution:
    query_text: str
    canonical_service: str


@dataclass(frozen=True)
class ServiceAwareDecision:
    detected_services: tuple[ServiceResolution, ...]
    ambiguous_aliases: tuple[str, ...]
    metadata_mode: MetadataMode
    filter_services: tuple[str, ...]
    boost_services: tuple[str, ...]
    reason: str

    @classmethod
    def global_decision(cls, *, reason: str) -> ServiceAwareDecision:
        return cls(
            detected_services=(),
            ambiguous_aliases=(),
            metadata_mode=MetadataMode.GLOBAL,
            filter_services=(),
            boost_services=(),
            reason=reason,
        )

    def detected_service_names(self) -> tuple[str, ...]:
        return tuple(sorted({detected.canonical_service for detected in self.detected_services}))


@dataclass(frozen=True)
class ServiceCatalogueEntry:
    source: str
    short_forms: tuple[str, ...]
    long_forms: tuple[str, ...]


ChunkT = TypeVar("ChunkT", bound="ScoredChunk")


class ScoredChunk(Protocol):
    chunk_id: str
    score: float
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ServiceAwareStrategyResult:
    chunks: list[ChunkT]
    strategy_error: str
    metadata_mode: str
    filter_services: tuple[str, ...]
    boost_services: tuple[str, ...]
    reason: str
    detected_services: tuple[str, ...]


class ServiceAwarePlanner:
    """Detects services and classifies retrieval mode from query wording."""

    def __init__(self, entries: tuple[ServiceCatalogueEntry, ...]) -> None:
        self.entries = entries
        alias_map: dict[str, set[str]] = {}
        for entry in entries:
            values = {entry.source, *entry.short_forms, *entry.long_forms}
            if not entry.source:
                continue
            for raw in values:
                normalized = normalize_query_token(raw)
                if normalized:
                    alias_map.setdefault(normalized, set()).add(entry.source)
        self.alias_map = alias_map
        self.aliases_by_length = tuple(sorted(alias_map.keys(), key=len, reverse=True))

    def plan(self, query_text: str) -> ServiceAwareDecision:
        expanded_query_text = expand_query_text(query_text, self.entries)
        detected, ambiguous = self._detect_services(expanded_query_text)
        metadata_mode, reason = self._classify_metadata_mode(
            query_text=query_text,
            detected=detected,
            ambiguous=ambiguous,
        )
        detected_services = tuple(detected.values())
        detected_names = tuple(sorted(detected))
        ambiguous_aliases = tuple(sorted(ambiguous))
        if metadata_mode == MetadataMode.HARD_FILTER:
            return ServiceAwareDecision(
                detected_services=detected_services,
                ambiguous_aliases=(),
                metadata_mode=metadata_mode,
                filter_services=(detected_services[0].canonical_service,),
                boost_services=(),
                reason=reason,
            )
        if metadata_mode == MetadataMode.BOOST:
            return ServiceAwareDecision(
                detected_services=detected_services,
                ambiguous_aliases=ambiguous_aliases,
                metadata_mode=metadata_mode,
                filter_services=(),
                boost_services=detected_names,
                reason=reason,
            )
        global_decision = ServiceAwareDecision.global_decision(reason=reason)
        return ServiceAwareDecision(
            detected_services=detected_services,
            ambiguous_aliases=ambiguous_aliases,
            metadata_mode=global_decision.metadata_mode,
            filter_services=global_decision.filter_services,
            boost_services=detected_names,
            reason=global_decision.reason,
        )

    def _detect_services(self, query_text: str) -> tuple[dict[str, ServiceResolution], set[str]]:
        query_normalized = normalize_query_token(query_text)
        detected: dict[str, ServiceResolution] = {}
        ambiguous: set[str] = set()

        for alias in self.aliases_by_length:
            if not alias:
                continue
            if not (
                phrase_present(query_normalized, alias)
                or phrase_present(query_text.lower(), alias)
            ):
                continue
            canonical_services = sorted(self.alias_map.get(alias, set()))
            if len(canonical_services) == 1:
                canonical = canonical_services[0]
                if canonical not in detected:
                    detected[canonical] = ServiceResolution(
                        query_text=alias,
                        canonical_service=canonical,
                    )
            elif canonical_services:
                ambiguous.add(alias)
        return detected, ambiguous

    def _classify_metadata_mode(
        self,
        *,
        query_text: str,
        detected: dict[str, ServiceResolution],
        ambiguous: set[str],
    ) -> tuple[MetadataMode, str]:
        if not self.alias_map:
            return MetadataMode.GLOBAL, "No service catalogue entries available; using global retrieval."
        detected_services = tuple(detected.values())
        service_discovery = self._is_service_discovery_query(query_text)

        if service_discovery or not detected_services:
            reason = (
                "No confident explicit service mention."
                if not detected_services
                else "Query wording requests service discovery/system-wide scope."
            )
            return MetadataMode.GLOBAL, reason

        if len(detected_services) == 1 and not ambiguous and self._is_explicitly_local_query(
            query_text=query_text,
            matched_service_text=detected_services[0].query_text,
        ):
            return (
                MetadataMode.HARD_FILTER,
                "Single explicit service mention with local-scope wording.",
            )

        return (
            MetadataMode.BOOST,
            "Ambiguous or non-local wording detected; using global retrieval with service-aware boosting.",
        )

    def _is_service_discovery_query(self, text: str) -> bool:
        lowered = text.lower()
        if "all services" in lowered or "across services" in lowered:
            return True
        return bool(re.search(r"\b(which|what)\s+(?:\w+\s+){0,3}services?\b", lowered))

    @staticmethod
    def _is_explicitly_local_query(*, query_text: str, matched_service_text: str) -> bool:
        lowered = query_text.lower()
        service = re.escape(matched_service_text.lower())
        if re.search(rf"\b(in|within|inside)\s+{service}\b", lowered):
            return True
        if re.search(rf"\b{service}\s*'s\b", lowered):
            return True
        return bool(re.search(rf"\bhow\s+does\s+{service}\b", lowered))


def resolve_strategy_decision(
    *,
    strategy: str,
    planned_decision: ServiceAwareDecision,
) -> tuple[str, ServiceAwareDecision]:
    token = strategy.strip().lower()
    if token.endswith(SERVICE_AWARE_SUFFIX):
        return token[: -len(SERVICE_AWARE_SUFFIX)], planned_decision
    return token, ServiceAwareDecision.global_decision(
        reason="Base strategy; service-aware routing disabled."
    )


def run_service_aware_strategy(
    *,
    search: Callable[[str, str, int, tuple[str, ...]], list[ChunkT]],
    base_strategy: str,
    query_text: str,
    top_k: int,
    decision: ServiceAwareDecision,
) -> ServiceAwareStrategyResult:
    try:
        chunks = search(base_strategy, query_text, top_k, decision.filter_services)
        if decision.metadata_mode == MetadataMode.BOOST:
            chunks = apply_service_boost(
                chunks=chunks,
                boost_services=decision.boost_services,
            )
        return ServiceAwareStrategyResult(
            chunks=chunks,
            strategy_error="",
            metadata_mode=decision.metadata_mode.value,
            filter_services=decision.filter_services,
            boost_services=decision.boost_services,
            reason=decision.reason,
            detected_services=decision.detected_service_names(),
        )
    except (RuntimeError, ValueError) as exc:
        return ServiceAwareStrategyResult(
            chunks=[],
            strategy_error=str(exc),
            metadata_mode=decision.metadata_mode.value,
            filter_services=decision.filter_services,
            boost_services=decision.boost_services,
            reason=decision.reason,
            detected_services=decision.detected_service_names(),
        )


def apply_service_boost(
    *,
    chunks: list[ChunkT],
    boost_services: tuple[str, ...],
) -> list[ChunkT]:
    if not chunks or not boost_services:
        return chunks
    target_services = {service.lower() for service in boost_services}
    rank_scores = {
        chunk.chunk_id: reciprocal_rank(rank)
        for rank, chunk in enumerate(chunks, start=1)
    }
    service_chunks = [
        chunk
        for chunk in chunks
        if _chunk_service_name(chunk) in target_services
    ]
    service_scores = {
        chunk.chunk_id: reciprocal_rank(rank)
        for rank, chunk in enumerate(service_chunks, start=1)
    }
    return sorted(
        chunks,
        key=lambda chunk: (
            rank_scores.get(chunk.chunk_id, 0.0)
            + service_scores.get(chunk.chunk_id, 0.0),
            chunk.score,
        ),
        reverse=True,
    )


def _chunk_service_name(chunk: ScoredChunk) -> str:
    value = chunk.metadata.get("source")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    if chunk.source.strip():
        return chunk.source.strip().lower()
    return ""


def build_service_planner(
    *,
    list_sources: Callable[[], tuple[str, ...]],
    service_catalogue_path: Path | None = None,
) -> ServiceAwarePlanner:
    if service_catalogue_path is not None:
        entries = load_service_catalogue(service_catalogue_path)
    else:
        entries = tuple(
            ServiceCatalogueEntry(
                source=source,
                short_forms=(source,),
                long_forms=(),
            )
            for source in list_sources()
        )
    return ServiceAwarePlanner(entries)


def load_service_catalogue(path: Path) -> tuple[ServiceCatalogueEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Service catalogue file must be a JSON list.")
    entries: list[ServiceCatalogueEntry] = []

    def _as_text_list(value: Any) -> list[str]:
        if isinstance(value, str):
            token = value.strip()
            return [token] if token else []
        if isinstance(value, list):
            tokens: list[str] = []
            for item in value:
                if isinstance(item, str):
                    token = item.strip()
                    if token:
                        tokens.append(token)
            return tokens
        return []

    for row in payload:
        if not isinstance(row, dict):
            continue
        sources = _as_text_list(row.get("source"))
        short_forms = _as_text_list(row.get("short_form"))
        long_forms = _as_text_list(row.get("long_form"))
        unique_sources = tuple(sorted(set(sources)))
        unique_short_forms = tuple(sorted(set(short_forms)))
        unique_long_forms = tuple(sorted(set(long_forms)))
        if not unique_short_forms and not unique_long_forms:
            continue
        if not unique_sources:
            entries.append(
                ServiceCatalogueEntry(
                    source="",
                    short_forms=unique_short_forms,
                    long_forms=unique_long_forms,
                )
            )
        for source in unique_sources:
            entries.append(
                ServiceCatalogueEntry(
                    source=source,
                    short_forms=unique_short_forms,
                    long_forms=unique_long_forms,
                )
            )
    return tuple(entries)
