"""Canonical graph seed resolution from query mentions."""

from __future__ import annotations

from typing import Any

from retrieval.query_processor import normalize_query_token

from .queries import build_seed_lookup_query
from .types import (
    SEEDABLE_NODE_LABELS,
    CypherSpec,
    EntityMention,
    GraphNodeRef,
    GraphReader,
    SeedMatch,
    SeedRequest,
    SeedResolution,
)

_MAX_SEED_MATCHES = 8


async def resolve_seeds(seed_request: SeedRequest, graph_reader: GraphReader) -> SeedResolution:
    matches: list[SeedMatch] = []
    unresolved: list[EntityMention] = []

    for mention in seed_request.mentions:
        found = await _lookup_for_mention(mention, graph_reader)

        if not found:
            unresolved.append(mention)
            continue

        for node in found:
            matches.append(
                SeedMatch(
                    mention=mention,
                    node=node,
                    match_score=mention.confidence * node.confidence,
                    match_reason=f"{mention.preferred_label or 'ANY'}-label lookup",
                )
            )

    deduped = dedupe_and_rank_seed_matches(matches)
    return SeedResolution(
        intent=seed_request.intent,
        seed_request=seed_request,
        matches=tuple(deduped),
        unresolved_mentions=tuple(unresolved),
    )


async def lookup_any_label(value: str, graph_reader: GraphReader) -> list[GraphNodeRef]:
    return await _lookup_with_label(value, None, graph_reader)


async def _lookup_for_mention(mention: EntityMention, graph_reader: GraphReader) -> list[GraphNodeRef]:
    if mention.preferred_label in SEEDABLE_NODE_LABELS:
        return await _lookup_with_label(mention.text, mention.preferred_label, graph_reader)
    return await lookup_any_label(mention.text, graph_reader)


async def _lookup_with_label(value: str, label: str | None, graph_reader: GraphReader) -> list[GraphNodeRef]:
    mention = EntityMention(text=value, normalized=normalize_query_token(value), preferred_label=label, confidence=1.0)
    return await _lookup_by_spec(graph_reader, build_seed_lookup_query(mention))


def dedupe_and_rank_seed_matches(matches: list[SeedMatch]) -> list[SeedMatch]:
    best_by_node: dict[str, SeedMatch] = {}
    for match in matches:
        existing = best_by_node.get(match.node.node_key)
        if existing is None or match.match_score > existing.match_score:
            best_by_node[match.node.node_key] = match
    ranked = sorted(
        best_by_node.values(),
        key=lambda item: (-item.match_score, -item.node.evidence_count, item.node.node_key),
    )
    return ranked[:_MAX_SEED_MATCHES]


async def _lookup_by_spec(graph_reader: GraphReader, spec: CypherSpec) -> list[GraphNodeRef]:
    rows = await graph_reader.fetch(spec.query, *spec.args)
    return [_row_to_node(row) for row in rows]


def _row_to_node(row: Any) -> GraphNodeRef:
    node_key = str(row["node_key"])
    node_label = str(row["node_label"])
    node_name = str(row["node_name"])
    confidence = float(row["confidence"])
    evidence_count = int(row["evidence_count"])
    return GraphNodeRef(
        node_key=node_key,
        node_label=node_label,
        node_name=node_name,
        confidence=confidence,
        evidence_count=evidence_count,
    )
