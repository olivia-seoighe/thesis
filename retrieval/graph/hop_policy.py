"""Adaptive traversal budgeting for graph retrieval."""

from __future__ import annotations

from collections import defaultdict

from .config import (
    ESCALATED_MARGINAL_GAIN_FLOOR,
    ESCALATION_EDGES_INCREMENT,
    ESCALATION_NODES_MULTIPLIER,
    ESCALATION_PATHS_MULTIPLIER,
    ESCALATION_CONFIDENCE_THRESHOLD,
    INITIAL_GENERAL_GLOBAL_NODE_BUDGET,
    INITIAL_GENERAL_GLOBAL_PATH_BUDGET,
    INITIAL_GENERAL_MAX_EDGES_PER_NODE,
    INITIAL_GENERAL_MAX_HOPS,
    INITIAL_GENERAL_MAX_LATENCY_MS,
    INITIAL_GENERAL_MAX_NODES_PER_HOP,
    INITIAL_LOCAL_GLOBAL_NODE_BUDGET,
    INITIAL_LOCAL_GLOBAL_PATH_BUDGET,
    INITIAL_LOCAL_MAX_EDGES_PER_NODE,
    INITIAL_LOCAL_MAX_HOPS,
    INITIAL_LOCAL_MAX_LATENCY_MS,
    INITIAL_LOCAL_MAX_NODES_PER_HOP,
    INITIAL_TOPOLOGY_GLOBAL_NODE_BUDGET,
    INITIAL_TOPOLOGY_GLOBAL_PATH_BUDGET,
    INITIAL_TOPOLOGY_MAX_EDGES_PER_NODE,
    INITIAL_TOPOLOGY_MAX_HOPS,
    INITIAL_TOPOLOGY_MAX_LATENCY_MS,
    INITIAL_TOPOLOGY_MAX_NODES_PER_HOP,
    MAX_EDGE_CAP,
    MAX_GLOBAL_NODE_CAP,
    MAX_GLOBAL_PATH_CAP,
    MAX_HOP_CAP,
    MAX_NODE_CAP,
    STOP_CONFIDENCE_FLOOR,
)
from .types import QueryIntent, SeedResolution, StopDecision, TraversalBudget, TraversalPlan, TraversalState


def plan_traversal(
    intent: QueryIntent,
    seed_resolution: SeedResolution,
    query: str,
    *,
    target_results: int = 5,
) -> TraversalPlan:
    del query
    seed_node_keys = tuple(match.node.node_key for match in seed_resolution.matches)
    return TraversalPlan(
        intent=intent,
        seed_node_keys=seed_node_keys,
        budget=initial_budgets(intent),
        target_results=max(1, target_results),
    )


def initial_budgets(intent: QueryIntent) -> TraversalBudget:
    if _is_global_intent(intent):
        return TraversalBudget(
            max_hops=INITIAL_TOPOLOGY_MAX_HOPS,
            max_nodes_per_hop=INITIAL_TOPOLOGY_MAX_NODES_PER_HOP,
            max_edges_per_node=INITIAL_TOPOLOGY_MAX_EDGES_PER_NODE,
            global_path_budget=INITIAL_TOPOLOGY_GLOBAL_PATH_BUDGET,
            global_node_budget=INITIAL_TOPOLOGY_GLOBAL_NODE_BUDGET,
            max_latency_ms=INITIAL_TOPOLOGY_MAX_LATENCY_MS,
        )
    if _is_local_intent(intent):
        return TraversalBudget(
            max_hops=INITIAL_LOCAL_MAX_HOPS,
            max_nodes_per_hop=INITIAL_LOCAL_MAX_NODES_PER_HOP,
            max_edges_per_node=INITIAL_LOCAL_MAX_EDGES_PER_NODE,
            global_path_budget=INITIAL_LOCAL_GLOBAL_PATH_BUDGET,
            global_node_budget=INITIAL_LOCAL_GLOBAL_NODE_BUDGET,
            max_latency_ms=INITIAL_LOCAL_MAX_LATENCY_MS,
        )
    return TraversalBudget(
        max_hops=INITIAL_GENERAL_MAX_HOPS,
        max_nodes_per_hop=INITIAL_GENERAL_MAX_NODES_PER_HOP,
        max_edges_per_node=INITIAL_GENERAL_MAX_EDGES_PER_NODE,
        global_path_budget=INITIAL_GENERAL_GLOBAL_PATH_BUDGET,
        global_node_budget=INITIAL_GENERAL_GLOBAL_NODE_BUDGET,
        max_latency_ms=INITIAL_GENERAL_MAX_LATENCY_MS,
    )


def should_escalate_depth(state: TraversalState, partial_results: list[object]) -> bool:
    if state.budget.max_hops >= MAX_HOP_CAP:
        return False
    if state.elapsed_ms >= state.budget.max_latency_ms:
        return False
    if state.escalations >= 3:
        return False
    trigger_count = 0
    if len(partial_results) < state.target_results:
        trigger_count += 1
    if state.avg_confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        trigger_count += 1
    if state.distinct_source_kinds <= 2:
        trigger_count += 1
    if state.top_hop_frontier_size < max(4, state.target_results):
        trigger_count += 1
    if state.newly_added_paths <= max(2, state.target_results // 3):
        trigger_count += 1
    if _is_local_intent(state.intent) and state.ast_path_count <= 1:
        trigger_count += 1
    return trigger_count >= 1


def _is_local_intent(intent: QueryIntent) -> bool:
    return intent == QueryIntent.LOCAL_LOGIC


def _is_global_intent(intent: QueryIntent) -> bool:
    return intent == QueryIntent.TOPOLOGY


def next_hop_budget(state: TraversalState, partial_results: list[object]) -> TraversalBudget:
    if not should_escalate_depth(state, partial_results):
        return state.budget
    return TraversalBudget(
        max_hops=min(MAX_HOP_CAP, state.budget.max_hops + 1),
        max_nodes_per_hop=min(
            MAX_NODE_CAP, int(state.budget.max_nodes_per_hop * ESCALATION_NODES_MULTIPLIER)
        ),
        max_edges_per_node=min(MAX_EDGE_CAP, state.budget.max_edges_per_node + ESCALATION_EDGES_INCREMENT),
        global_path_budget=min(
            MAX_GLOBAL_PATH_CAP, int(state.budget.global_path_budget * ESCALATION_PATHS_MULTIPLIER)
        ),
        global_node_budget=min(
            MAX_GLOBAL_NODE_CAP, int(state.budget.global_node_budget * ESCALATION_PATHS_MULTIPLIER)
        ),
        max_latency_ms=state.budget.max_latency_ms,
    )


def should_stop(state: TraversalState, partial_results: list[object]) -> StopDecision:
    if state.current_hop >= state.budget.max_hops:
        return StopDecision(True, "hop_limit_reached")
    if state.visited_nodes >= state.budget.global_node_budget:
        return StopDecision(True, "node_budget_reached")
    if state.visited_paths >= state.budget.global_path_budget:
        return StopDecision(True, "path_budget_reached")
    if state.elapsed_ms >= state.budget.max_latency_ms:
        return StopDecision(True, "latency_budget_exceeded")
    if state.newly_added_paths <= ESCALATED_MARGINAL_GAIN_FLOOR and state.escalations > 0:
        return StopDecision(True, "marginal_gain_low")
    if state.avg_confidence < STOP_CONFIDENCE_FLOOR:
        return StopDecision(True, "frontier_confidence_floor")
    if len(partial_results) >= state.target_results and state.current_hop >= min(2, state.budget.max_hops):
        return StopDecision(True, "results_sufficient")
    return StopDecision(False, "continue")


def prune_frontier_rows(
    rows: list[dict],
    *,
    intent: QueryIntent,
    seed_node_keys: tuple[str, ...],
    budget: TraversalBudget,
) -> list[dict]:
    del intent
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("hop", 0) or 0),
            -float(row.get("confidence", 0.0) or 0.0),
            str(row.get("subject_key", "")),
            str(row.get("object_key", "")),
        ),
    )

    kept: list[dict] = []
    global_nodes = set(seed_node_keys)
    hop_nodes: dict[int, set[str]] = defaultdict(set)
    edges_per_subject: dict[str, int] = defaultdict(int)

    for row in sorted_rows:
        hop = int(row.get("hop", 0) or 0)
        if hop <= 0 or hop > budget.max_hops:
            continue

        subject_key = str(row.get("subject_key", ""))
        object_key = str(row.get("object_key", ""))
        if not subject_key or not object_key:
            continue

        if edges_per_subject[subject_key] >= budget.max_edges_per_node:
            continue

        introduces_new_global = subject_key not in global_nodes or object_key not in global_nodes
        if introduces_new_global and len(global_nodes) >= budget.global_node_budget:
            continue

        new_hop_nodes = [node for node in (subject_key, object_key) if node not in hop_nodes[hop]]
        if new_hop_nodes and len(hop_nodes[hop]) + len(new_hop_nodes) > budget.max_nodes_per_hop:
            continue

        kept.append(row)
        edges_per_subject[subject_key] += 1
        global_nodes.add(subject_key)
        global_nodes.add(object_key)
        hop_nodes[hop].add(subject_key)
        hop_nodes[hop].add(object_key)

        if len(kept) >= budget.global_path_budget:
            break

    return kept
