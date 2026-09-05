"""Graph candidate ranking utilities."""

from __future__ import annotations

import math
import re
from typing import Any

from .lexicon import COMMON_QUERY_STOPWORDS
from .types import EvidenceBundle, GraphPath, NodeCandidate, QueryIntent, RankedChunk, RankingContext

_EVIDENCE_COUNT_CAP = 5
_EVIDENCE_BONUS_DIVISOR = 20
_LINE_BOUNDS_BONUS = 0.05
_MAX_PROVENANCE_MULTIPLIER = 1.3
_QUERY_COVERAGE_WEIGHT = 0.3
_EXACT_NODE_MATCH_BONUS = 0.2

# Scores one traversed graph path for a candidate chunk.
def _score_path(path: GraphPath, context: RankingContext) -> tuple[float, dict[str, Any]]:
    relation_weight = _relation_weight(path.predicates, context)
    hop_penalty = compute_hop_penalty(path.hops, context.decay_lambda)
    provenance_bonus = compute_provenance_bonus(path.evidence)
    score = path.confidence * relation_weight * hop_penalty * provenance_bonus
    diagnostics = {
        "path_confidence": path.confidence,
        "relation_weight": relation_weight,
        "hop_penalty": hop_penalty,
        "path_provenance_bonus": provenance_bonus,
        "path_hops": path.hops,
        "path_predicates": list(path.predicates),
        "path_tiers": sorted(set(path.evidence.tiers)),
        "path_score": score,
    }
    return score, diagnostics


# Scores one candidate chunk from graph path and evidence signals.
def _score_candidate(
    candidate: NodeCandidate,
    supporting_paths: list[GraphPath],
    context: RankingContext,
) -> tuple[float, dict[str, Any]]:
    top_path_diag: dict[str, Any] = {}
    if supporting_paths:
        scored_paths = [_score_path(path, context) for path in supporting_paths]
        path_component, top_path_diag = max(scored_paths, key=lambda item: item[0])
    else:
        path_component = candidate.path_score

    evidence_component = compute_provenance_bonus(candidate.evidence)
    query_relevance_multiplier, query_relevance_diag = _query_relevance_multiplier(
        candidate,
        context.query_terms,
    )
    final_score = path_component * evidence_component * query_relevance_multiplier
    diagnostics = {
        "intent": context.intent,
        "node_key": candidate.node.node_key,
        "node_label": candidate.node.node_label,
        "seedless_path_component": path_component,
        "candidate_provenance_bonus": evidence_component,
        "query_relevance_multiplier": query_relevance_multiplier,
        "query_relevance": query_relevance_diag,
        "candidate_tiers": sorted(candidate.evidence.tiers),
        "candidate_source_kinds": list(candidate.evidence.source_kinds),
        "supporting_path_count": len(supporting_paths),
        "final_score": final_score,
    }
    if top_path_diag:
        diagnostics["top_path"] = top_path_diag
    return final_score, diagnostics


# Applies an exponential penalty as graph paths move farther from seeds.
def compute_hop_penalty(hops: int, decay_lambda: float) -> float:
    safe_hops = max(1, hops)
    return math.exp(-decay_lambda * (safe_hops - 1))


# Boosts candidates supported by multiple evidence rows or precise lines.
def compute_provenance_bonus(evidence: EvidenceBundle) -> float:
    evidence_bonus = min(evidence.evidence_count, _EVIDENCE_COUNT_CAP) / _EVIDENCE_BONUS_DIVISOR
    line_bonus = _LINE_BOUNDS_BONUS if evidence.has_line_bounds else 0.0
    return min(_MAX_PROVENANCE_MULTIPLIER, 1.0 + evidence_bonus + line_bonus)


# Converts scored graph candidates into deduplicated ranked chunks.
def aggregate_to_chunks(
    candidates: list[NodeCandidate],
    mapping: dict[str, list[GraphPath]],
    top_k: int,
    context: RankingContext | None = None,
) -> list[RankedChunk]:
    ranking_context = context or RankingContext(intent=QueryIntent.GENERAL)
    scored: list[RankedChunk] = []
    for candidate in candidates:
        path_list = mapping.get(candidate.node.node_key, [])
        score, score_diagnostics = _score_candidate(candidate, path_list, ranking_context)
        scored.append(
            RankedChunk(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_title=candidate.document_title,
                text=candidate.text,
                source_code=candidate.source_code,
                source=candidate.source,
                url=candidate.url,
                last_modified_date=candidate.last_modified_date,
                metadata=candidate.metadata,
                score=score,
                score_diagnostics=score_diagnostics,
            )
        )

    deduped: dict[str, RankedChunk] = {}
    for chunk in scored:
        existing = deduped.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            deduped[chunk.chunk_id] = chunk

    ranked = sorted(
        deduped.values(),
        key=lambda item: (-item.score, item.document_id, item.chunk_id),
    )
    return ranked[: max(top_k, 0)]


def _relation_weight(predicates: tuple[str, ...], context: RankingContext) -> float:
    if not predicates:
        return 1.0
    effective_weights = context.relation_weights
    weights = [effective_weights.get(predicate, 1.0) for predicate in predicates]
    return sum(weights) / len(weights)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,}", re.IGNORECASE)


def _query_relevance_multiplier(
    candidate: NodeCandidate,
    query_terms: tuple[str, ...],
) -> tuple[float, dict[str, Any]]:
    if not query_terms:
        return 1.0, {"matched_terms": [], "coverage": 0.0}
    query_term_set = set(query_terms)
    text_blob = " ".join((candidate.node.node_name, candidate.document_title, candidate.text[:800])).lower()
    matched = [term for term in query_terms if term in text_blob]
    coverage = len(matched) / max(len(query_terms), 1)
    exact_node_match = candidate.node.node_name.lower() in query_term_set
    multiplier = 1.0 + min(coverage, 1.0) * _QUERY_COVERAGE_WEIGHT
    if exact_node_match:
        multiplier += _EXACT_NODE_MATCH_BONUS
    return multiplier, {
        "matched_terms": matched,
        "coverage": round(coverage, 4),
        "exact_node_match": exact_node_match,
    }


def extract_query_terms(query: str) -> tuple[str, ...]:
    terms = [token.lower() for token in _TOKEN_RE.findall(query)]
    filtered = [
        token
        for token in terms
        if len(token) > 2 and token not in COMMON_QUERY_STOPWORDS and not token.isdigit()
    ]
    deduped = []
    seen: set[str] = set()
    for token in filtered:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return tuple(deduped[:20])
