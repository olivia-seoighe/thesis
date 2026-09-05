"""Structured graph-query routing for aggregate retrieval questions."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from retrieval.models.models import RetrievedChunk, SearchRequest, SearchResponse
from retrieval.utils.logging_config import get_logger

from .config import STRUCTURED_QUERY_MIN_SIMILARITY
from .entity_linker import _repo_seed_sources, build_query_seed_mentions, build_seed_candidates
from .queries import (
    build_known_repos_query,
    build_structured_distinct_pairs_query,
    build_structured_evidence_rows_query,
)
from .seed_resolver import resolve_seeds
from .types import API_LABEL, FRAMEWORK_LABEL, NUGET_PACKAGE_LABEL, REPO_LABEL, TOPIC_LABEL

logger = get_logger(__name__)


class RoutingShape(StrEnum):
    SUBJECT = "SUBJECT"
    SUBJECT_ALL = "SUBJECT_ALL"
    EXHAUSTIVE = "EXHAUSTIVE"
    TOPIC_RELATED = "TOPIC_RELATED"


@dataclass(frozen=True)
class _Template:
    template_id: str
    text: str
    predicate: str
    routing_shape: RoutingShape
    anchor_labels: tuple[str, ...]


# Reference phrases for the embedding classifier.
_TEMPLATES: tuple[_Template, ...] = (
    _Template("subject_package_1", "Which services use the package?", "CONTAINS_PACKAGE", RoutingShape.SUBJECT, (NUGET_PACKAGE_LABEL,)),
    _Template("subject_package_2", "Is this library used anywhere across the services?", "CONTAINS_PACKAGE", RoutingShape.SUBJECT, (NUGET_PACKAGE_LABEL,)),
    _Template("subject_api_1", "Which services call this API?", "CALLS_API", RoutingShape.SUBJECT, (API_LABEL,)),
    _Template("subject_api_2", "Which services send requests to this API, and what endpoint do they use?", "CALLS_API", RoutingShape.SUBJECT, (API_LABEL,)),
    _Template("subject_exposes_api", "Which services expose this API?", "EXPOSES_API", RoutingShape.SUBJECT, (API_LABEL,)),
    _Template("subject_framework", "Which services target this framework?", "TARGETS_FRAMEWORK", RoutingShape.SUBJECT, (FRAMEWORK_LABEL,)),
    _Template("subject_all_saga", "Which services implement sagas, and how many does each have?", "OWNS_SAGA", RoutingShape.SUBJECT_ALL, ()),
    _Template("subject_all_consumes", "Which services consume any Kafka topics at all?", "CONSUMES_TOPIC", RoutingShape.SUBJECT_ALL, ()),
    _Template("subject_all_produces", "Which services publish any Kafka topics at all?", "PRODUCES_TOPIC", RoutingShape.SUBJECT_ALL, ()),
    _Template("exhaustive_package_1", "Which services are missing this package?", "CONTAINS_PACKAGE", RoutingShape.EXHAUSTIVE, (NUGET_PACKAGE_LABEL,)),
    _Template("exhaustive_package_2", "Which services handle this concern a different way, instead of using this package?", "CONTAINS_PACKAGE", RoutingShape.EXHAUSTIVE, (NUGET_PACKAGE_LABEL,)),
    _Template("exhaustive_api", "Which services don't call this API?", "CALLS_API", RoutingShape.EXHAUSTIVE, (API_LABEL,)),
    _Template("exhaustive_framework", "Which services haven't migrated to this framework yet?", "TARGETS_FRAMEWORK", RoutingShape.EXHAUSTIVE, (FRAMEWORK_LABEL,)),
    _Template("topic_related_1", "Which service publishes this topic, and which other services consume it?", "PRODUCES_TOPIC", RoutingShape.TOPIC_RELATED, (REPO_LABEL, TOPIC_LABEL)),
    _Template("topic_related_2", "Who produces and who consumes this topic?", "PRODUCES_TOPIC", RoutingShape.TOPIC_RELATED, (REPO_LABEL, TOPIC_LABEL)),
    _Template("topic_related_3", "Which services are involved in producing or consuming this topic?", "PRODUCES_TOPIC", RoutingShape.TOPIC_RELATED, (REPO_LABEL, TOPIC_LABEL)),
    _Template("topic_related_4", "What events does this service publish, and who listens for them?", "PRODUCES_TOPIC", RoutingShape.TOPIC_RELATED, (REPO_LABEL, TOPIC_LABEL)),
    _Template("topic_related_5", "What does this service publish downstream, and which services consume it?", "PRODUCES_TOPIC", RoutingShape.TOPIC_RELATED, (REPO_LABEL, TOPIC_LABEL)),
)

_ABSENCE_FALLBACK_PREDICATE = "TARGETS_FRAMEWORK"
_DIRECT_MATCH_SCORE = 1.0
_ABSENCE_FALLBACK_SCORE = 0.4


@dataclass(frozen=True)
class _ClassifiedTemplate:
    template_id: str
    predicate: str
    routing_shape: RoutingShape
    anchor_labels: tuple[str, ...]
    similarity: float


_template_embeddings_cache: list[list[float]] | None = None
_template_embeddings_lock = asyncio.Lock()


async def _get_template_embeddings(embedding_client: Any) -> list[list[float]]:
    global _template_embeddings_cache
    if _template_embeddings_cache is not None:
        return _template_embeddings_cache
    async with _template_embeddings_lock:
        if _template_embeddings_cache is None:
            _template_embeddings_cache = await asyncio.gather(*(embedding_client.embed_single(t.text) for t in _TEMPLATES))
    return _template_embeddings_cache


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def classify_structured_query(query_embedding: list[float], embedding_client: Any) -> _ClassifiedTemplate | None:
    template_embeddings = await _get_template_embeddings(embedding_client)
    best: _ClassifiedTemplate | None = None
    for template, embedding in zip(_TEMPLATES, template_embeddings):
        similarity = _cosine(query_embedding, embedding)
        if best is None or similarity > best.similarity:
            best = _ClassifiedTemplate(
                template_id=template.template_id,
                predicate=template.predicate,
                routing_shape=template.routing_shape,
                anchor_labels=template.anchor_labels,
                similarity=similarity,
            )
    if best is None or best.similarity < STRUCTURED_QUERY_MIN_SIMILARITY:
        return None
    return best


async def maybe_route_structured_query(
    request: SearchRequest,
    *,
    search_client: Any,
    service_aliases: list[tuple[str, list[str]]],
    start_time: float,
) -> SearchResponse | None:
    try:
        embedding_client = getattr(search_client, "embedding_client", None)
        if embedding_client is None:
            return None

        query_embedding = await embedding_client.embed_single(request.query)
        classified = await classify_structured_query(query_embedding, embedding_client)
        if classified is None:
            return None

        anchor_object_keys: list[str] | None = None
        anchor_repo_names: list[str] | None = None
        if classified.anchor_labels:
            query_for_seeding, mentions = build_query_seed_mentions(request.query, service_aliases)
            seed_request = build_seed_candidates(query_for_seeding, mentions, source_filters=request.sources)
            seed_resolution = await resolve_seeds(seed_request, search_client, embedding_client=embedding_client)

            if REPO_LABEL in classified.anchor_labels:
                repo_names = _repo_seed_sources(seed_resolution.matches)
                if repo_names:
                    anchor_repo_names = sorted(repo_names)

            other_matches = [
                m for m in seed_resolution.matches
                if m.node.node_label in classified.anchor_labels and m.node.node_label != REPO_LABEL
            ]
            if other_matches:
                anchor_object_keys = sorted({m.node.node_key for m in other_matches})

            if not anchor_repo_names and not anchor_object_keys:
                return None

        rows = await _execute_shape(
            search_client,
            predicate=classified.predicate,
            routing_shape=classified.routing_shape,
            anchor_object_keys=anchor_object_keys,
            anchor_repo_names=anchor_repo_names,
            source_filter=list(request.sources) or None,
            retrieval_corpus=request.retrieval_corpus,
        )
        if not rows:
            return None

        chunks = _rows_to_chunks(rows, classified=classified)[: request.top_k]
        return SearchResponse(
            chunks=chunks,
            total_results=len(chunks),
            search_duration_ms=(time.time() - start_time) * 1000,
            embedding_duration_ms=0.0,
            model_used="graph-structured-router-v1",
            source_searched=",".join(request.sources) if request.sources else "all",
        )
    except Exception:
        logger.warning("Structured query routing failed, falling back to normal traversal", exc_info=True)
        return None


def _intersect(repos: set[str], source_filter: list[str] | None) -> set[str]:
    if not source_filter:
        return repos
    allowed = {s.lower() for s in source_filter}
    return {repo for repo in repos if repo.lower() in allowed}


async def _fetch(search_client: Any, spec) -> list[dict[str, Any]]:
    rows = await search_client.fetch(spec.query, *spec.args)
    return [dict(row) for row in rows]


async def _execute_shape(
    search_client: Any,
    *,
    predicate: str,
    routing_shape: RoutingShape,
    anchor_object_keys: list[str] | None,
    anchor_repo_names: list[str] | None,
    source_filter: list[str] | None,
    retrieval_corpus: str,
) -> list[dict[str, Any]]:
    if routing_shape == RoutingShape.SUBJECT:
        return await _shape_subject(search_client, predicate=predicate, object_keys=anchor_object_keys, source_filter=source_filter, retrieval_corpus=retrieval_corpus)
    if routing_shape == RoutingShape.SUBJECT_ALL:
        return await _shape_subject(search_client, predicate=predicate, object_keys=None, source_filter=source_filter, retrieval_corpus=retrieval_corpus)
    if routing_shape == RoutingShape.EXHAUSTIVE:
        return await _shape_exhaustive(search_client, predicate=predicate, object_keys=anchor_object_keys, source_filter=source_filter, retrieval_corpus=retrieval_corpus)
    if routing_shape == RoutingShape.TOPIC_RELATED:
        if anchor_repo_names:
            return await _shape_topic_producer_consumers(search_client, repo_names=anchor_repo_names, source_filter=source_filter, retrieval_corpus=retrieval_corpus)
        if anchor_object_keys:
            return await _shape_topic_both(search_client, object_keys=anchor_object_keys, source_filter=source_filter, retrieval_corpus=retrieval_corpus)
        return []
    return []


async def _shape_subject(search_client, *, predicate, object_keys, source_filter, retrieval_corpus) -> list[dict[str, Any]]:
    pairs = await _fetch(search_client, build_structured_distinct_pairs_query(predicate=predicate, object_keys=object_keys))
    repos = _intersect({str(r["source_repo"]) for r in pairs}, source_filter)
    if not repos:
        return []
    rows = await _fetch(
        search_client,
        build_structured_evidence_rows_query(predicate=predicate, object_keys=object_keys, source_repos=sorted(repos), retrieval_corpus=retrieval_corpus),
    )
    for row in rows:
        row["_absence"] = False
    return rows


async def _shape_exhaustive(search_client, *, predicate, object_keys, source_filter, retrieval_corpus) -> list[dict[str, Any]]:
    matched_rows = await _fetch(
        search_client,
        build_structured_evidence_rows_query(predicate=predicate, object_keys=object_keys, source_repos=None, retrieval_corpus=retrieval_corpus),
    )
    for row in matched_rows:
        row["_absence"] = False
    matched_repos = {row["source_repo"] for row in matched_rows}

    known_repos = {str(r["source"]) for r in await _fetch(search_client, build_known_repos_query())}
    missing_repos = known_repos - matched_repos

    result = list(matched_rows)
    if missing_repos:
        fallback_rows = await _fetch(
            search_client,
            build_structured_evidence_rows_query(
                predicate=_ABSENCE_FALLBACK_PREDICATE, object_keys=None, source_repos=sorted(missing_repos), retrieval_corpus=retrieval_corpus
            ),
        )
        for row in fallback_rows:
            row["_absence"] = True
        result.extend(fallback_rows)

    kept_repos = _intersect({row["source_repo"] for row in result}, source_filter)
    return [row for row in result if row["source_repo"] in kept_repos]


async def _shape_topic_both(search_client, *, object_keys, source_filter, retrieval_corpus) -> list[dict[str, Any]]:
    produces_pairs = await _fetch(search_client, build_structured_distinct_pairs_query(predicate="PRODUCES_TOPIC", object_keys=object_keys))
    consumes_pairs = await _fetch(search_client, build_structured_distinct_pairs_query(predicate="CONSUMES_TOPIC", object_keys=object_keys))
    repos = _intersect(
        {str(r["source_repo"]) for r in produces_pairs} | {str(r["source_repo"]) for r in consumes_pairs},
        source_filter,
    )
    if not repos:
        return []
    result: list[dict[str, Any]] = []
    for predicate in ("PRODUCES_TOPIC", "CONSUMES_TOPIC"):
        rows = await _fetch(
            search_client,
            build_structured_evidence_rows_query(predicate=predicate, object_keys=object_keys, source_repos=sorted(repos), retrieval_corpus=retrieval_corpus),
        )
        for row in rows:
            row["_absence"] = False
        result.extend(rows)
    return result


async def _shape_topic_producer_consumers(search_client, *, repo_names, source_filter, retrieval_corpus) -> list[dict[str, Any]]:
    if not repo_names:
        return []
    produced_pairs = await _fetch(search_client, build_structured_distinct_pairs_query(predicate="PRODUCES_TOPIC", source_repos=repo_names))
    produced_keys = sorted({str(r["object_key"]) for r in produced_pairs})
    if not produced_keys:
        return []
    consumer_pairs = await _fetch(search_client, build_structured_distinct_pairs_query(predicate="CONSUMES_TOPIC", object_keys=produced_keys))
    consumer_repos = {str(r["source_repo"]) for r in consumer_pairs} - set(repo_names)

    result = await _fetch(
        search_client,
        build_structured_evidence_rows_query(predicate="PRODUCES_TOPIC", source_repos=repo_names, retrieval_corpus=retrieval_corpus),
    )
    for row in result:
        row["_absence"] = False
    if consumer_repos:
        consumer_rows = await _fetch(
            search_client,
            build_structured_evidence_rows_query(
                predicate="CONSUMES_TOPIC", object_keys=produced_keys, source_repos=sorted(consumer_repos), retrieval_corpus=retrieval_corpus
            ),
        )
        for row in consumer_rows:
            row["_absence"] = False
        result.extend(consumer_rows)

    kept_repos = _intersect({row["source_repo"] for row in result}, source_filter)
    return [row for row in result if row["source_repo"] in kept_repos]


def _coerce_metadata(value: Any) -> dict[str, Any]:
    # Mirrors GraphClient._coerce_metadata: jsonb columns aren't always
    # auto-decoded to dicts by asyncpg depending on how the value reaches the
    # driver (e.g. via a COALESCE(...::jsonb) cast), so handle both shapes.
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _rows_to_chunks(rows: list[dict[str, Any]], *, classified: _ClassifiedTemplate) -> list[RetrievedChunk]:
    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            bool(row.get("_absence")),
            -float(row.get("confidence") or 0.0),
            str(row["source_repo"]),
            str(row["source_path"]),
        )

    seen: set[tuple[str, str]] = set()
    chunks: list[RetrievedChunk] = []
    for row in sorted(rows, key=sort_key):
        key = (str(row["source_repo"]), str(row["source_path"]))
        if key in seen:
            continue
        seen.add(key)
        is_absence = bool(row.get("_absence"))
        metadata = _coerce_metadata(row.get("metadata"))
        metadata["structured_route"] = {
            "template_id": classified.template_id,
            "predicate": classified.predicate,
            "routing_shape": classified.routing_shape.value,
            "similarity": classified.similarity,
            "absence_fallback": is_absence,
        }
        chunks.append(
            RetrievedChunk(
                chunk_id=str(row["resolved_chunk_id"]),
                text=str(row["text"]),
                document_id=str(row["document_id"]),
                document_title=str(row["document_title"]),
                last_modified_date=str(row["last_modified_date"] or ""),
                score=_ABSENCE_FALLBACK_SCORE if is_absence else _DIRECT_MATCH_SCORE,
                metadata=metadata,
                source=str(row["source"] or ""),
                url=str(row["url"] or ""),
                source_code=str(row["source_code"] or ""),
            )
        )
    return chunks
