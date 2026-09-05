"""Graph seed resolution from query mentions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from retrieval.query_processor import normalize_query_token
from retrieval.utils.logging_config import get_logger

from .config import (
    EMBEDDING_SEED_MIN_SIMILARITY,
    EMBEDDING_SEED_TIMEOUT_SECONDS,
    EMBEDDING_SEED_TOP_K_PER_LABEL,
    MAX_SEED_MATCHES,
)
from .entity_linker import is_generic_label_term
from .queries import build_embedding_seed_lookup_query, build_label_seed_lookup_query, build_seed_lookup_query
from .types import (
    SEEDABLE_NODE_LABELS,
    CypherSpec,
    EmbeddingClientLike,
    EntityMention,
    GraphNodeRef,
    GraphReader,
    SeedMatch,
    SeedRequest,
    SeedResolution,
)

logger = get_logger(__name__)


async def resolve_seeds(
    seed_request: SeedRequest,
    graph_reader: GraphReader,
    *,
    embedding_client: EmbeddingClientLike | None = None,
) -> SeedResolution:
    matches: list[SeedMatch] = []
    unresolved: list[EntityMention] = []

    for mention in seed_request.mentions:
        found = await _lookup_for_mention(mention, graph_reader)
        if not found:
            unresolved.append(mention)
        matches.extend(found)

    if embedding_client is not None:
        matches.extend(await resolve_embedding_seeds(seed_request.query, graph_reader, embedding_client))

    deduped = dedupe_and_rank_seed_matches(matches)
    return SeedResolution(
        intent=seed_request.intent,
        seed_request=seed_request,
        matches=tuple(deduped),
        unresolved_mentions=tuple(unresolved),
    )


REASON_ANY_LABEL = "ANY-label lookup"


async def _lookup_for_mention(mention: EntityMention, graph_reader: GraphReader) -> list[SeedMatch]:
    if mention.preferred_label not in SEEDABLE_NODE_LABELS:
        nodes = await lookup_any_label(mention.text, graph_reader)
        return _wrap_matches(mention, nodes, reason=REASON_ANY_LABEL)

    if is_generic_label_term(label=mention.preferred_label, normalized_term=mention.normalized):
        return []

    direct = await _lookup_with_label(mention.text, mention.preferred_label, graph_reader)
    if direct:
        return _wrap_matches(mention, direct, reason=f"{mention.preferred_label}-direct")

    bulk = await _lookup_by_label(mention.preferred_label, graph_reader)
    return _wrap_matches(mention, bulk, reason=f"{mention.preferred_label}-label-scan")


def _wrap_matches(mention: EntityMention, nodes: list[GraphNodeRef], *, reason: str) -> list[SeedMatch]:
    return [
        SeedMatch(mention=mention, node=node, match_score=mention.confidence * node.confidence, match_reason=reason)
        for node in nodes
    ]


async def resolve_embedding_seeds(
    query: str,
    graph_reader: GraphReader,
    embedding_client: EmbeddingClientLike,
    *,
    top_k_per_label: int = EMBEDDING_SEED_TOP_K_PER_LABEL,
    min_similarity: float = EMBEDDING_SEED_MIN_SIMILARITY,
) -> list[SeedMatch]:
    """Resolve additional seed candidates with graph node-name embeddings."""
    try:

        async def _fetch() -> list[Any]:
            query_embedding = await embedding_client.embed_single(query)
            spec = build_embedding_seed_lookup_query(
                query_embedding=query_embedding,
                labels=sorted(SEEDABLE_NODE_LABELS),
                top_k_per_label=top_k_per_label,
                min_similarity=min_similarity,
            )
            return await graph_reader.fetch(spec.query, *spec.args)

        rows = await asyncio.wait_for(_fetch(), timeout=EMBEDDING_SEED_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("Embedding-based seed lookup failed, skipping", exc_info=True)
        return []

    matches: list[SeedMatch] = []
    for row in rows:
        node = _row_to_node(row)
        mention = EntityMention(
            text=query,
            normalized=f"embedding::{node.node_label}",
            preferred_label=node.node_label,
            confidence=float(row["similarity"]),
        )
        matches.extend(_wrap_matches(mention, [node], reason="embedding-similarity"))
    return matches


async def lookup_any_label(value: str, graph_reader: GraphReader) -> list[GraphNodeRef]:
    return await _lookup_with_label(value, None, graph_reader)


async def _lookup_with_label(value: str, label: str | None, graph_reader: GraphReader) -> list[GraphNodeRef]:
    mention = EntityMention(text=value, normalized=normalize_query_token(value), preferred_label=label, confidence=1.0)
    return await _lookup_by_spec(graph_reader, build_seed_lookup_query(mention))


async def _lookup_by_label(label: str, graph_reader: GraphReader) -> list[GraphNodeRef]:
    return await _lookup_by_spec(graph_reader, build_label_seed_lookup_query(label=label))


def dedupe_and_rank_seed_matches(matches: list[SeedMatch]) -> list[SeedMatch]:
    best_by_node: dict[str, SeedMatch] = {}
    for match in matches:
        existing = best_by_node.get(match.node.node_key)
        if existing is None or match.match_score > existing.match_score:
            best_by_node[match.node.node_key] = match

    mention_groups: dict[tuple[str, str | None], list[SeedMatch]] = defaultdict(list)
    for match in best_by_node.values():
        key = (match.mention.normalized, match.mention.preferred_label)
        mention_groups[key].append(match)

    for grouped_matches in mention_groups.values():
        grouped_matches.sort(
            key=lambda item: (-item.match_score, -item.node.evidence_count, item.node.node_key),
        )

    selected: list[SeedMatch] = []
    labeled_groups = {key: group for key, group in mention_groups.items() if key[1]}
    unlabeled_groups = {key: group for key, group in mention_groups.items() if not key[1]}

    _select_round_robin(selected, labeled_groups, limit=MAX_SEED_MATCHES)
    if len(selected) < MAX_SEED_MATCHES:
        _select_round_robin(selected, unlabeled_groups, limit=MAX_SEED_MATCHES)

    return selected


def _select_round_robin(
    selected: list[SeedMatch],
    groups: dict[tuple[str, str | None], list[SeedMatch]],
    *,
    limit: int,
) -> None:
    ordered_group_keys = sorted(
        groups,
        key=lambda key: (
            -max(item.match_score for item in groups[key]),
            -len(groups[key]),
            key[0],
            key[1] or "",
        ),
    )
    round_index = 0
    while len(selected) < limit:
        picked_in_round = False
        for group_key in ordered_group_keys:
            group = groups[group_key]
            if round_index >= len(group):
                continue
            selected.append(group[round_index])
            picked_in_round = True
            if len(selected) >= limit:
                return
        if not picked_in_round:
            return
        round_index += 1


async def _lookup_by_spec(graph_reader: GraphReader, spec: CypherSpec) -> list[GraphNodeRef]:
    rows = await graph_reader.fetch(spec.query, *spec.args)
    return [_row_to_node(row) for row in rows]


def _row_to_node(row: Any) -> GraphNodeRef:
    return GraphNodeRef(
        node_key=str(row["node_key"]),
        node_label=str(row["node_label"]),
        node_name=str(row["node_name"]),
        confidence=float(row["confidence"]),
        evidence_count=int(row["evidence_count"]),
    )
