"""Graph candidate ranking utilities."""

from __future__ import annotations

import math
import re
from typing import Any

from .config import (
    RANK_PROVENANCE_EVIDENCE_WEIGHT,
    RANK_PROVENANCE_LINE_BOUNDS_BONUS,
    RANK_PROVENANCE_MAX_MULTIPLIER,
    RANK_PROVENANCE_SOURCE_DIVERSITY_WEIGHT,
    RANK_QUERY_BASE_MULTIPLIER,
    RANK_QUERY_COVERAGE_WEIGHT,
    RANK_QUERY_EXACT_NODE_BONUS,
    RANK_TOPOLOGY_MESSAGING_DOC_BONUS,
)
from .lexicon import COMMON_QUERY_STOPWORDS
from .types import EvidenceBundle, GraphPath, NodeCandidate, QueryIntent, RankedChunk, RankingContext


# 1) PATH SCORING -------------------------------------------------------------
def compute_path_scoring(path: GraphPath, context: RankingContext) -> tuple[float, dict[str, Any]]:
    policy = compute_ranking_policy(
        intent=context.intent,
        tiers=set(path.evidence.tiers),
        predicates=path.predicates,
        context=context,
    )
    hop_penalty = compute_hop_penalty(path.hops, context.decay_lambda)
    provenance_bonus = compute_provenance_bonus(path.evidence)
    weighted_base = path.confidence * policy["relation_weight"]
    weighted = weighted_base * policy["tier_weight"]
    score = weighted * hop_penalty * provenance_bonus
    diagnostics = {
        "path_confidence": path.confidence,
        "relation_weight": policy["relation_weight"],
        "tier_weight": policy["tier_weight"],
        "hop_penalty": hop_penalty,
        "path_provenance_bonus": provenance_bonus,
        "path_hops": path.hops,
        "path_predicates": list(path.predicates),
        "path_tiers": sorted(set(path.evidence.tiers)),
        "path_score": score,
    }
    return score, diagnostics


def score_path(path: GraphPath, context: RankingContext) -> float:
    score, _ = compute_path_scoring(path, context)
    return score


# 2) CANDIDATE SCORING --------------------------------------------------------
def compute_candidate_scoring(
    candidate: NodeCandidate,
    supporting_paths: list[GraphPath],
    context: RankingContext,
) -> tuple[float, dict[str, Any]]:
    top_path_diag: dict[str, Any] = {}
    if supporting_paths:
        scored_paths = [compute_path_scoring(path, context) for path in supporting_paths]
        path_component, top_path_diag = max(scored_paths, key=lambda item: item[0])
    else:
        path_component = candidate.path_score

    evidence_component = compute_provenance_bonus(candidate.evidence)
    candidate_tiers = set(candidate.evidence.tiers)
    policy = compute_ranking_policy(
        intent=context.intent,
        tiers=candidate_tiers,
        predicates=(),
        context=context,
    )
    query_relevance_multiplier, query_relevance_diag = _query_relevance_multiplier(
        candidate,
        context.query_terms,
        context.intent,
    )
    final_score = (
        path_component
        * evidence_component
        * policy["tier_weight"]
        * policy["anti_skew_multiplier"]
        * query_relevance_multiplier
    )
    diagnostics = {
        "intent": context.intent,
        "node_key": candidate.node.node_key,
        "node_label": candidate.node.node_label,
        "seedless_path_component": path_component,
        "candidate_provenance_bonus": evidence_component,
        "candidate_tier_weight": policy["tier_weight"],
        "anti_skew_multiplier": policy["anti_skew_multiplier"],
        "query_relevance_multiplier": query_relevance_multiplier,
        "query_relevance": query_relevance_diag,
        "candidate_tiers": sorted(candidate_tiers),
        "candidate_source_kinds": list(candidate.evidence.source_kinds),
        "supporting_path_count": len(supporting_paths),
        "final_score": final_score,
    }
    if top_path_diag:
        diagnostics["top_path"] = top_path_diag
    return final_score, diagnostics


def score_node_candidate(
    candidate: NodeCandidate,
    supporting_paths: list[GraphPath],
    context: RankingContext,
) -> tuple[float, dict[str, Any]]:
    return compute_candidate_scoring(candidate, supporting_paths, context)


def compute_hop_penalty(hops: int, decay_lambda: float) -> float:
    safe_hops = max(1, hops)
    return math.exp(-decay_lambda * (safe_hops - 1))


def compute_provenance_bonus(evidence: EvidenceBundle) -> float:
    b1 = RANK_PROVENANCE_EVIDENCE_WEIGHT * min(evidence.evidence_count, 5)
    b2 = RANK_PROVENANCE_SOURCE_DIVERSITY_WEIGHT * min(max(len(evidence.source_kinds) - 1, 0), 3)
    b3 = RANK_PROVENANCE_LINE_BOUNDS_BONUS if evidence.has_line_bounds else 0.0
    return min(RANK_PROVENANCE_MAX_MULTIPLIER, 1 + b1 + b2 + b3)


def apply_intent_tier_weights(base_score: float, intent: QueryIntent, tiers: set[str]) -> float:
    return base_score * _intent_tier_multiplier(intent, tiers)


# 3) RANKING POLICY -----------------------------------------------------------
def compute_ranking_policy(
    *,
    intent: QueryIntent,
    tiers: set[str],
    predicates: tuple[str, ...],
    context: RankingContext,
) -> dict[str, float]:
    return {
        "relation_weight": _relation_weight(predicates, context),
        "tier_weight": _intent_tier_multiplier(intent, tiers),
        "anti_skew_multiplier": _anti_skew_multiplier(intent, tiers),
    }


def _intent_tier_multiplier(intent: QueryIntent, tiers: set[str]) -> float:
    has_ast = "AST_LOCAL" in tiers
    has_contract = "CONTRACT_GLOBAL" in tiers
    if intent == QueryIntent.LOCAL_LOGIC:
        if has_ast and has_contract:
            return 1.08
        if has_ast:
            return 1.05
        if has_contract:
            return 0.78
    if intent == QueryIntent.TOPOLOGY:
        if has_contract:
            return 1.08
        if has_ast:
            return 0.84
    return 1.0


def _anti_skew_multiplier(intent: QueryIntent, tiers: set[str]) -> float:
    has_ast = "AST_LOCAL" in tiers
    has_contract = "CONTRACT_GLOBAL" in tiers
    if intent == QueryIntent.LOCAL_LOGIC and has_contract and not has_ast:
        return 0.82
    if intent == QueryIntent.TOPOLOGY and has_ast and not has_contract:
        return 0.8
    return 1.0


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
        score, score_diagnostics = score_node_candidate(candidate, path_list, ranking_context)
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
    intent: QueryIntent,
) -> tuple[float, dict[str, Any]]:
    if not query_terms:
        return 1.0, {"matched_terms": [], "coverage": 0.0}
    query_term_set = set(query_terms)
    text_blob = " ".join((candidate.node.node_name, candidate.document_title, candidate.text[:800])).lower()
    matched = [term for term in query_terms if term in text_blob]
    coverage = len(matched) / max(len(query_terms), 1)
    exact_node_match = candidate.node.node_name.lower() in query_term_set
    multiplier = (
        RANK_QUERY_BASE_MULTIPLIER
        + min(coverage, 1.0) * RANK_QUERY_COVERAGE_WEIGHT
        + (RANK_QUERY_EXACT_NODE_BONUS if exact_node_match else 0.0)
    )
    title = candidate.document_title.lower()
    messaging_terms = {"kafka", "topic", "topics", "publish", "consume", "consumes", "producer", "consumer"}
    if intent == QueryIntent.TOPOLOGY and messaging_terms.intersection(query_term_set):
        if "appsettings" in title or "servicecollection" in title:
            multiplier += RANK_TOPOLOGY_MESSAGING_DOC_BONUS
    return multiplier, {
        "matched_terms": matched,
        "coverage": round(coverage, 4),
        "exact_node_match": exact_node_match,
        "intent": str(intent),
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
