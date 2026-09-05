"""Traversal budgeting for graph retrieval."""

from __future__ import annotations

import os
from collections import defaultdict

from .types import QueryIntent, SeedResolution, TraversalBudget, TraversalPlan


# Selects one bounded graph traversal budget for each query intent.
GRAPH_TRAVERSAL_BUDGETS: dict[QueryIntent, TraversalBudget] = {
    QueryIntent.TOPOLOGY: TraversalBudget(
        max_hops=int(os.getenv("GRAPH_TOPOLOGY_MAX_HOPS", "2")),
        max_nodes_per_hop=int(os.getenv("GRAPH_TOPOLOGY_MAX_NODES_PER_HOP", "60")),
        max_edges_per_node=int(os.getenv("GRAPH_TOPOLOGY_MAX_EDGES_PER_NODE", "8")),
        global_path_budget=int(os.getenv("GRAPH_TOPOLOGY_PATH_BUDGET", "320")),
        global_node_budget=int(os.getenv("GRAPH_TOPOLOGY_NODE_BUDGET", "320")),
    ),
    QueryIntent.LOCAL_LOGIC: TraversalBudget(
        max_hops=int(os.getenv("GRAPH_LOCAL_MAX_HOPS", "3")),
        max_nodes_per_hop=int(os.getenv("GRAPH_LOCAL_MAX_NODES_PER_HOP", "34")),
        max_edges_per_node=int(os.getenv("GRAPH_LOCAL_MAX_EDGES_PER_NODE", "5")),
        global_path_budget=int(os.getenv("GRAPH_LOCAL_PATH_BUDGET", "220")),
        global_node_budget=int(os.getenv("GRAPH_LOCAL_NODE_BUDGET", "220")),
    ),
    QueryIntent.GENERAL: TraversalBudget(
        max_hops=int(os.getenv("GRAPH_GENERAL_MAX_HOPS", "2")),
        max_nodes_per_hop=int(os.getenv("GRAPH_GENERAL_MAX_NODES_PER_HOP", "48")),
        max_edges_per_node=int(os.getenv("GRAPH_GENERAL_MAX_EDGES_PER_NODE", "6")),
        global_path_budget=int(os.getenv("GRAPH_GENERAL_PATH_BUDGET", "240")),
        global_node_budget=int(os.getenv("GRAPH_GENERAL_NODE_BUDGET", "240")),
    ),
}


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
    return GRAPH_TRAVERSAL_BUDGETS.get(intent, GRAPH_TRAVERSAL_BUDGETS[QueryIntent.GENERAL])


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
