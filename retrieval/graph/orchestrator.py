"""Graph retrieval orchestrator."""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

from .config import RANK_DECAY
from .entity_linker import (
    build_query_seed_mentions,
    build_seed_candidates,
    infer_intent_from_seed_labels,
    resolve_effective_sources,
    resolve_graph_scope,
)
from .hop_policy import plan_traversal, prune_frontier_rows
from .queries import build_evidence_query, build_frontier_expansion_query
from .ranker import aggregate_to_chunks, extract_query_terms
from .seed_resolver import resolve_seeds
from .structured_router import maybe_route_structured_query
from .types import (
    EvidenceBundle,
    GraphScope,
    GraphNodeRef,
    GraphPath,
    NodeCandidate,
    QueryIntent,
    RankedChunk,
    RankingContext,
    SeedMatch,
    SeedResolution,
    GraphTraversalMeta,
    TraversalPlan,
)
from retrieval.models.models import RetrievedChunk, SearchRequest, SearchResponse
from retrieval.strategies.service_aware import load_service_catalogue
from retrieval.utils.logging_config import get_logger

logger = get_logger(__name__)


class GraphClient:
    """Retrieves chunks by linking query mentions to graph-backed evidence."""

    def __init__(self, connection_manager, service_catalogue_path: Path | None = None) -> None:
        self.cm = connection_manager
        self._service_aliases = self._load_service_aliases(service_catalogue_path)

    async def structured_search(self, request: SearchRequest, *, start_time: float | None = None) -> SearchResponse | None:
        return await maybe_route_structured_query(
            request,
            search_client=self.cm,
            service_aliases=self._service_aliases,
            start_time=start_time if start_time is not None else time.time(),
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.time()

        structured_response = await self.structured_search(request, start_time=start_time)
        if structured_response is not None:
            return structured_response

        seed_start = time.time()
        query_for_seeding, mentions = build_query_seed_mentions(request.query, self._service_aliases)
        seed_request = build_seed_candidates(
            query_for_seeding,
            mentions,
            source_filters=request.sources,
        )

        seed_resolution = await resolve_seeds(
            seed_request, self.cm, embedding_client=getattr(self.cm, "embedding_client", None)
        )
        seed_ms = (time.time() - seed_start) * 1000
        graph_scope = resolve_graph_scope(
            query=request.query,
            explicit_sources=request.sources,
            mentions=seed_request.mentions,
            seed_matches=seed_resolution.matches,
        )
        effective_sources = resolve_effective_sources(
            explicit_sources=request.sources,
            mentions=seed_request.mentions,
            seed_matches=seed_resolution.matches,
            graph_scope=graph_scope,
        )
        if not seed_resolution.matches:
            logger.info(
                "Graph search returned no seed matches",
                extra={"graph_traversal_meta": {"intent": seed_request.intent, "seed_ms": seed_ms}},
            )
            return self._empty_response(request=request, start_time=start_time)

        resolved_seed_labels = {match.node.node_label for match in seed_resolution.matches}
        effective_intent = infer_intent_from_seed_labels(
            resolved_seed_labels,
            fallback_intent=seed_request.intent,
        )
        if effective_intent != seed_request.intent:
            logger.info(
                "Graph intent reclassified from resolved seed labels",
                extra={
                    "graph_traversal_meta": {
                        "initial_intent": seed_request.intent,
                        "effective_intent": effective_intent,
                        "resolved_seed_labels": sorted(resolved_seed_labels),
                    }
                },
            )

        traversal = plan_traversal(
            intent=effective_intent,
            seed_resolution=seed_resolution,
            query=request.query,
            target_results=request.top_k,
        )
        traversal_result = await self._traverse_with_policy(traversal)
        neighborhood_rows = traversal_result["rows"]
        traversal_meta: GraphTraversalMeta = traversal_result["meta"]

        candidate_node_keys = self._collect_candidate_node_keys(seed_resolution, neighborhood_rows, traversal)
        if not candidate_node_keys:
            logger.info("Graph search produced no candidate node keys", extra={"graph_traversal_meta": traversal_meta.__dict__})
            return self._empty_response(request=request, start_time=start_time)

        evidence_start = time.time()
        evidence_spec = build_evidence_query(candidate_node_keys, request.retrieval_corpus)
        evidence_rows = await self.cm.fetch(evidence_spec.query, *evidence_spec.args)
        evidence_ms = (time.time() - evidence_start) * 1000

        ranking_start = time.time()
        candidates = self._build_candidates(
            evidence_rows,
            effective_sources,
            intent=effective_intent,
            seed_node_labels={match.node.node_label for match in seed_resolution.matches},
            mention_label_hints={mention.preferred_label for mention in seed_request.mentions if mention.preferred_label},
        )
        if not candidates:
            logger.info("Graph search produced no ranked candidates", extra={"graph_traversal_meta": traversal_meta.__dict__})
            return self._empty_response(request=request, start_time=start_time)

        path_mapping = self._build_path_mapping(neighborhood_rows, seed_resolution.matches)
        candidate_pool_size = request.top_k
        if effective_intent == QueryIntent.TOPOLOGY and graph_scope == GraphScope.GLOBAL:
            candidate_pool_size = min(200, max(request.top_k * 6, request.top_k))

        ranked = aggregate_to_chunks(
            candidates,
            path_mapping,
            candidate_pool_size,
            RankingContext(
                intent=effective_intent,
                decay_lambda=RANK_DECAY,
                query_terms=extract_query_terms(request.query),
            ),
        )
        if effective_intent == QueryIntent.TOPOLOGY and graph_scope == GraphScope.GLOBAL:
            ranked = self._apply_service_diversity(ranked, request.top_k, strict=True)
        ranked = self._apply_document_diversity(ranked, request.max_chunks_per_document)
        ranking_ms = (time.time() - ranking_start) * 1000
        total_ms = (time.time() - start_time) * 1000
        traversal_meta = self._update_timing(
            traversal_meta,
            seed_ms=seed_ms,
            evidence_ms=evidence_ms,
            ranking_ms=ranking_ms,
            total_ms=total_ms,
        )
        logger.info("Graph traversal completed", extra={"graph_traversal_meta": traversal_meta.__dict__})

        return SearchResponse(
            chunks=[
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    last_modified_date=chunk.last_modified_date,
                    score=chunk.score,
                    metadata=self._with_graph_diagnostics(chunk, traversal_meta),
                    source=chunk.source,
                    url=chunk.url,
                    source_code=chunk.source_code,
                )
                for chunk in ranked
            ],
            total_results=len(ranked),
            search_duration_ms=total_ms,
            embedding_duration_ms=0.0,
            model_used="graph-traversal-v1",
            source_searched=",".join(effective_sources) if effective_sources else "all",
        )

    async def _traverse_with_policy(
        self,
        traversal: TraversalPlan,
    ) -> dict[str, Any]:
        traversal_start = time.time()
        rows = await self._expand_with_budget(traversal)
        rows = prune_frontier_rows(
            rows,
            intent=traversal.intent,
            seed_node_keys=traversal.seed_node_keys,
            budget=traversal.budget,
        )
        meta = self._build_traversal_meta(
            traversal=traversal,
            rows=rows,
            stop_reason=self._traversal_stop_reason(traversal, rows),
            traversal_start=traversal_start,
        )
        return {"rows": rows, "meta": meta}

    async def _expand_with_budget(self, traversal: TraversalPlan) -> list[dict[str, Any]]:
        frontier = set(traversal.seed_node_keys)
        visited_nodes = set(traversal.seed_node_keys)
        seen_edge_keys: set[str] = set()
        rows: list[dict[str, Any]] = []

        for hop in range(1, traversal.budget.max_hops + 1):
            if not frontier:
                break

            remaining_budget = traversal.budget.global_path_budget - len(rows)
            if remaining_budget <= 0:
                break

            per_hop_limit = len(frontier) * traversal.budget.max_edges_per_node
            edge_limit = max(10, min(remaining_budget, per_hop_limit))
            spec = build_frontier_expansion_query(
                intent=traversal.intent,
                frontier_node_keys=sorted(frontier),
                edge_limit=edge_limit,
            )
            fetched_rows = [dict(row) for row in await self.cm.fetch(spec.query, *spec.args)]
            if not fetched_rows:
                break

            next_frontier: set[str] = set()
            frontier_nodes = set(frontier)
            for row in fetched_rows:
                subject_key = str(row["subject_key"])
                object_key = str(row["object_key"])
                edge_key = str(row["edge_key"])
                if not subject_key or not object_key or not edge_key:
                    continue
                if edge_key in seen_edge_keys:
                    continue

                seen_edge_keys.add(edge_key)
                row["hop"] = hop
                rows.append(row)
                if len(rows) >= traversal.budget.global_path_budget:
                    break

                if subject_key in frontier_nodes and object_key not in visited_nodes:
                    next_frontier.add(object_key)
                if object_key in frontier_nodes and subject_key not in visited_nodes:
                    next_frontier.add(subject_key)

            visited_nodes.update(next_frontier)
            frontier = next_frontier
            if len(rows) >= traversal.budget.global_path_budget:
                break

        return rows

    @staticmethod
    def _collect_candidate_node_keys(
        seed_resolution: SeedResolution,
        neighborhood_rows: list[dict[str, Any]],
        traversal: TraversalPlan,
    ) -> list[str]:
        keys = {match.node.node_key for match in seed_resolution.matches}
        for row in neighborhood_rows:
            keys.add(str(row["subject_key"]))
            keys.add(str(row["object_key"]))
            if len(keys) >= traversal.budget.global_node_budget:
                break
        return sorted(keys)

    @staticmethod
    def _build_candidates(
        rows: list[Any],
        source_filters: list[str],
        *,
        intent: QueryIntent,
        seed_node_labels: set[str],
        mention_label_hints: set[str],
    ) -> list[NodeCandidate]:
        source_filter_set = {value.lower() for value in source_filters}
        candidates: list[NodeCandidate] = []
        code_focused_labels = {"SAGA", "HANDLER", "COMMAND", "EVENT"} & seed_node_labels
        explicit_code_focus_labels = code_focused_labels & mention_label_hints
        code_focused_local = intent == QueryIntent.LOCAL_LOGIC and bool(explicit_code_focus_labels)
        allowed_code_focused_labels = explicit_code_focus_labels

        for row in rows:
            source = str(row["source"] or "")
            if source_filter_set and source.lower() not in source_filter_set:
                continue
            node_label = str(row["node_label"])
            if intent == QueryIntent.LOCAL_LOGIC and code_focused_local and node_label not in allowed_code_focused_labels:
                continue

            evidence = EvidenceBundle(
                evidence_count=int(row["evidence_count"]),
                source_kinds=tuple(sorted(str(item) for item in row["source_kinds"])),
                has_line_bounds=bool(row["has_line_bounds"]),
                tiers=tuple(sorted(str(item) for item in row["tiers"])),
            )
            candidates.append(
                NodeCandidate(
                    node=GraphNodeRef(
                        node_key=str(row["node_key"]),
                        node_label=str(row["node_label"]),
                        node_name=str(row["node_name"]),
                        confidence=float(row["max_confidence"]),
                        evidence_count=int(row["evidence_count"]),
                    ),
                    document_id=str(row["document_id"]),
                    chunk_id=str(row["resolved_chunk_id"]),
                    source=source,
                    document_title=str(row["document_title"]),
                    text=str(row["text"]),
                    source_code=str(row["source_code"] or ""),
                    metadata=GraphClient._coerce_metadata(row["metadata"]),
                    url=str(row["url"] or ""),
                    last_modified_date=str(row["last_modified_date"] or ""),
                    evidence=evidence,
                    path_score=float(row["max_confidence"]),
                )
            )

        return candidates

    @staticmethod
    def _coerce_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _build_path_mapping(
        neighborhood_rows: list[dict[str, Any]],
        seed_matches: tuple[SeedMatch, ...],
    ) -> dict[str, list[GraphPath]]:
        mapping: dict[str, list[GraphPath]] = {}
        seed_scores = {match.node.node_key: match.match_score for match in seed_matches}

        for row in neighborhood_rows:
            subject_key = str(row["subject_key"])
            object_key = str(row["object_key"])
            confidence = float(row["confidence"])
            hop = int(row["hop"])
            evidence = EvidenceBundle(
                evidence_count=1,
                source_kinds=(str(row["source_kind"]),),
                has_line_bounds=row["line_start"] is not None and row["line_end"] is not None,
                tiers=(str(row["tier"]),),
            )
            path = GraphPath(
                start_node_key=subject_key,
                end_node_key=object_key,
                predicates=(str(row["predicate"]),),
                hops=hop,
                confidence=max(confidence, seed_scores.get(subject_key, 0.0)),
                evidence=evidence,
            )
            mapping.setdefault(subject_key, []).append(path)
            mapping.setdefault(object_key, []).append(path)

        return mapping

    @staticmethod
    def _apply_document_diversity(
        ranked: list[RankedChunk],
        max_chunks_per_document: int | None,
    ) -> list[RankedChunk]:
        if max_chunks_per_document is None:
            return ranked
        per_doc: dict[str, int] = {}
        filtered = []
        for chunk in ranked:
            count = per_doc.get(chunk.document_id, 0)
            if count >= max_chunks_per_document:
                continue
            per_doc[chunk.document_id] = count + 1
            filtered.append(chunk)
        return filtered

    @staticmethod
    def _apply_service_diversity(
        ranked: list[RankedChunk],
        top_k: int,
        *,
        strict: bool = False,
    ) -> list[RankedChunk]:
        if top_k <= 1:
            return ranked
        selected: list[RankedChunk] = []
        seen_buckets: set[str] = set()

        service_ranked = [chunk for chunk in ranked if GraphClient._service_bucket(chunk.document_title).startswith("src:")]
        other_ranked = [chunk for chunk in ranked if not GraphClient._service_bucket(chunk.document_title).startswith("src:")]

        for chunk in service_ranked:
            bucket = GraphClient._service_bucket(chunk.document_title)
            if bucket in seen_buckets:
                continue
            selected.append(chunk)
            seen_buckets.add(bucket)
            if len(selected) >= top_k:
                return selected

        if strict and selected:
            return selected

        for chunk in other_ranked:
            bucket = GraphClient._service_bucket(chunk.document_title)
            if bucket in seen_buckets:
                continue
            selected.append(chunk)
            seen_buckets.add(bucket)
            if len(selected) >= top_k:
                return selected

        if strict:
            return selected

        if len(selected) >= top_k:
            return selected

        selected_ids = {chunk.chunk_id for chunk in selected}
        for chunk in ranked:
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _service_bucket(document_title: str) -> str:
        title = (document_title or "").strip().lower()
        if not title:
            return "unknown"
        parts = title.split("/")
        if len(parts) >= 2 and parts[0] == "src":
            return f"src:{GraphClient._normalize_service_folder(parts[1])}"
        if parts[0] == "k8s":
            return "k8s"
        return f"path:{title}"

    @staticmethod
    def _normalize_service_folder(folder: str) -> str:
        tokens = [token for token in folder.lower().split(".") if token]
        suffix_tokens = {
            "api",
            "core",
            "di",
            "messages",
            "service",
            "svc",
            "webapi",
            "worker",
        }
        while len(tokens) > 1 and tokens[-1] in suffix_tokens:
            tokens.pop()
        return ".".join(tokens) if tokens else folder.lower()

    def _load_service_aliases(self, service_catalogue_path: Path | None) -> list[tuple[str, list[str]]]:
        catalogue_path = service_catalogue_path or (Path(__file__).resolve().parents[2] / "service_acronyms.json")
        if not catalogue_path.exists():
            return []

        entries = load_service_catalogue(catalogue_path)
        aliases: list[tuple[str, list[str]]] = []
        for entry in entries:
            all_aliases = sorted({*entry.short_forms, *entry.long_forms})
            aliases.append((entry.source, all_aliases))
        return aliases

    @staticmethod
    def _with_graph_diagnostics(chunk: RankedChunk, traversal_meta: GraphTraversalMeta) -> dict[str, Any]:
        metadata = dict(chunk.metadata or {})
        metadata["graph_score_debug"] = chunk.score_diagnostics
        metadata["graph_traversal_meta"] = traversal_meta.__dict__
        return metadata

    @staticmethod
    def _empty_response(*, request: SearchRequest, start_time: float) -> SearchResponse:
        return SearchResponse(
            chunks=[],
            total_results=0,
            search_duration_ms=(time.time() - start_time) * 1000,
            embedding_duration_ms=0.0,
            model_used="graph-traversal-v1",
            source_searched=",".join(request.sources) if request.sources else "all",
        )

    @staticmethod
    def _budget_dict(budget) -> dict[str, int]:
        return {
            "max_hops": int(budget.max_hops),
            "max_nodes_per_hop": int(budget.max_nodes_per_hop),
            "max_edges_per_node": int(budget.max_edges_per_node),
            "global_path_budget": int(budget.global_path_budget),
            "global_node_budget": int(budget.global_node_budget),
        }

    def _build_traversal_meta(
        self,
        *,
        traversal: TraversalPlan,
        rows: list[dict[str, Any]],
        stop_reason: str,
        traversal_start: float,
    ) -> GraphTraversalMeta:
        frontier_sizes: dict[str, int] = {}
        for row in rows:
            hop = str(int(row.get("hop", 0) or 0))
            frontier_sizes[hop] = frontier_sizes.get(hop, 0) + 1
        node_keys = {key for row in rows for key in (row["subject_key"], row["object_key"])}
        node_keys.update(traversal.seed_node_keys)
        timing = {
            "seed_ms": 0.0,
            "traversal_ms": (time.time() - traversal_start) * 1000,
            "evidence_ms": 0.0,
            "ranking_ms": 0.0,
            "total_ms": 0.0,
        }
        return GraphTraversalMeta(
            intent=str(traversal.intent),
            initial_budget=self._budget_dict(traversal.budget),
            final_budget=self._budget_dict(traversal.budget),
            hops_executed=max((int(row.get("hop", 0) or 0) for row in rows), default=0),
            frontier_sizes_by_hop=frontier_sizes,
            nodes_visited=len(node_keys),
            paths_examined=len(rows),
            stop_reason=stop_reason,
            timing_ms=timing,
        )

    @staticmethod
    def _update_timing(
        traversal_meta: GraphTraversalMeta,
        *,
        seed_ms: float,
        evidence_ms: float,
        ranking_ms: float,
        total_ms: float,
    ) -> GraphTraversalMeta:
        timing = dict(traversal_meta.timing_ms)
        timing["seed_ms"] = seed_ms
        timing["evidence_ms"] = evidence_ms
        timing["ranking_ms"] = ranking_ms
        timing["total_ms"] = total_ms
        return GraphTraversalMeta(
            intent=traversal_meta.intent,
            initial_budget=traversal_meta.initial_budget,
            final_budget=traversal_meta.final_budget,
            hops_executed=traversal_meta.hops_executed,
            frontier_sizes_by_hop=traversal_meta.frontier_sizes_by_hop,
            nodes_visited=traversal_meta.nodes_visited,
            paths_examined=traversal_meta.paths_examined,
            stop_reason=traversal_meta.stop_reason,
            timing_ms=timing,
        )

    @staticmethod
    def _traversal_stop_reason(traversal: TraversalPlan, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "no_paths_found"
        if len(rows) >= traversal.budget.global_path_budget:
            return "path_budget_reached"
        hops_executed = max((int(row.get("hop", 0) or 0) for row in rows), default=0)
        if hops_executed >= traversal.budget.max_hops:
            return "hop_limit_reached"
        return "budget_complete"
