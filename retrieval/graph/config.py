"""Configuration defaults for graph retrieval behavior.

Tuning guide:
- Budget/cap constants trade retrieval recall vs latency.
- Threshold constants control when traversal escalates or stops.
- Ranking constants control final ordering strength of provenance and query relevance.
"""

from __future__ import annotations

import os
from typing import Final

GRAPH_NAME: Final[str] = os.getenv("GRAPH_NAME", "enterprise_graph")  # Must match ingestion/indexing graph name so Cypher targets the same graph.

SEED_LOOKUP_LIMIT: Final[int] = int(os.getenv("GRAPH_SEED_LOOKUP_LIMIT", "15"))  # Caps seed fan-out so ambiguous mentions do not explode traversal.
TOPOLOGY_QUERY_LIMIT: Final[int] = int(os.getenv("GRAPH_TOPOLOGY_QUERY_LIMIT", "500"))  # Keeps topology endpoint bounded for latency and payload size.

MAX_HOP_CAP: Final[int] = int(os.getenv("GRAPH_MAX_HOP_CAP", "4"))  # Prevents deep multi-hop drift into weakly related neighborhoods.
MAX_NODE_CAP: Final[int] = int(os.getenv("GRAPH_MAX_NODE_CAP", "180"))  # Hard upper bound on per-hop breadth after escalation.
MAX_EDGE_CAP: Final[int] = int(os.getenv("GRAPH_MAX_EDGE_CAP", "16"))  # Limits per-node edge expansion to avoid high-degree hubs dominating.
MAX_GLOBAL_PATH_CAP: Final[int] = int(os.getenv("GRAPH_MAX_GLOBAL_PATH_CAP", "520"))  # Absolute cap on accumulated traversed paths in one query.
MAX_GLOBAL_NODE_CAP: Final[int] = int(os.getenv("GRAPH_MAX_GLOBAL_NODE_CAP", "520"))  # Absolute cap on discovered nodes in one query.
ESCALATION_NODES_MULTIPLIER: Final[float] = float(
    os.getenv("GRAPH_ESCALATION_NODES_MULTIPLIER", "1.5")
)  # Expands node budget by 50% when escalation is justified.
ESCALATION_PATHS_MULTIPLIER: Final[float] = float(
    os.getenv("GRAPH_ESCALATION_PATHS_MULTIPLIER", "1.35")
)  # Expands global path/node budgets moderately to trade recall for extra latency.
ESCALATION_EDGES_INCREMENT: Final[int] = int(
    os.getenv("GRAPH_ESCALATION_EDGES_INCREMENT", "3")
)  # Widens per-node frontier gradually instead of removing limits entirely.

INITIAL_TOPOLOGY_MAX_HOPS: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_MAX_HOPS", "2")
)  # Topology questions usually resolve within short structural paths.
INITIAL_TOPOLOGY_MAX_NODES_PER_HOP: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_MAX_NODES_PER_HOP", "60")
)  # Allows broader service-neighborhood sampling for cross-service topology.
INITIAL_TOPOLOGY_MAX_EDGES_PER_NODE: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_MAX_EDGES_PER_NODE", "8")
)  # Higher than local logic to traverse shared infra connectors.
INITIAL_TOPOLOGY_GLOBAL_PATH_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_GLOBAL_PATH_BUDGET", "320")
)  # Initial cap tuned to capture broad topology without runaway traversal.
INITIAL_TOPOLOGY_GLOBAL_NODE_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_GLOBAL_NODE_BUDGET", "320")
)  # Mirrors path budget for balanced breadth growth.
INITIAL_TOPOLOGY_MAX_LATENCY_MS: Final[int] = int(
    os.getenv("GRAPH_INIT_TOPOLOGY_MAX_LATENCY_MS", "1600")
)  # Target latency budget for topology mode under normal load.

INITIAL_FLOW_MAX_HOPS: Final[int] = int(os.getenv("GRAPH_INIT_FLOW_MAX_HOPS", "3"))  # Flow tracing needs one extra hop for producer/consumer chains.
INITIAL_FLOW_MAX_NODES_PER_HOP: Final[int] = int(
    os.getenv("GRAPH_INIT_FLOW_MAX_NODES_PER_HOP", "42")
)  # Narrower breadth than topology to stay focused on causal chains.
INITIAL_FLOW_MAX_EDGES_PER_NODE: Final[int] = int(
    os.getenv("GRAPH_INIT_FLOW_MAX_EDGES_PER_NODE", "6")
)  # Restrains branching while preserving alternate route discovery.
INITIAL_FLOW_GLOBAL_PATH_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_FLOW_GLOBAL_PATH_BUDGET", "280")
)  # Balanced cap for recall on flows without topology-style breadth.
INITIAL_FLOW_GLOBAL_NODE_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_FLOW_GLOBAL_NODE_BUDGET", "280")
)  # Matches path scale for consistent pruning behavior.
INITIAL_FLOW_MAX_LATENCY_MS: Final[int] = int(
    os.getenv("GRAPH_INIT_FLOW_MAX_LATENCY_MS", "1800")
)  # Slightly higher latency allowance due to deeper default hops.

INITIAL_LOCAL_MAX_HOPS: Final[int] = int(os.getenv("GRAPH_INIT_LOCAL_MAX_HOPS", "3"))  # Local logic often spans handler->command/event chains.
INITIAL_LOCAL_MAX_NODES_PER_HOP: Final[int] = int(
    os.getenv("GRAPH_INIT_LOCAL_MAX_NODES_PER_HOP", "34")
)  # Tightest breadth to avoid drifting away from service-local code context.
INITIAL_LOCAL_MAX_EDGES_PER_NODE: Final[int] = int(
    os.getenv("GRAPH_INIT_LOCAL_MAX_EDGES_PER_NODE", "5")
)  # Conservative edge fan-out for explainable local reasoning paths.
INITIAL_LOCAL_GLOBAL_PATH_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_LOCAL_GLOBAL_PATH_BUDGET", "220")
)  # Smaller budget reflects narrower expected result space.
INITIAL_LOCAL_GLOBAL_NODE_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_LOCAL_GLOBAL_NODE_BUDGET", "220")
)  # Keeps node growth proportional to local path budget.
INITIAL_LOCAL_MAX_LATENCY_MS: Final[int] = int(
    os.getenv("GRAPH_INIT_LOCAL_MAX_LATENCY_MS", "1800")
)  # Similar to flow due to same hop depth and AST-heavy filtering.

INITIAL_CONFIG_MAX_HOPS: Final[int] = int(os.getenv("GRAPH_INIT_CONFIG_MAX_HOPS", "2"))  # Config questions are usually direct dependency lookups.
INITIAL_CONFIG_MAX_NODES_PER_HOP: Final[int] = int(
    os.getenv("GRAPH_INIT_CONFIG_MAX_NODES_PER_HOP", "50")
)  # Moderate breadth to cover environment/config fan-out.
INITIAL_CONFIG_MAX_EDGES_PER_NODE: Final[int] = int(
    os.getenv("GRAPH_INIT_CONFIG_MAX_EDGES_PER_NODE", "7")
)  # Slightly elevated for config hubs that legitimately connect multiple targets.
INITIAL_CONFIG_GLOBAL_PATH_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_CONFIG_GLOBAL_PATH_BUDGET", "280")
)  # Sized between topology and local logic for mixed breadth/precision.
INITIAL_CONFIG_GLOBAL_NODE_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_CONFIG_GLOBAL_NODE_BUDGET", "280")
)  # Parity with path budget keeps pruning predictable.
INITIAL_CONFIG_MAX_LATENCY_MS: Final[int] = int(
    os.getenv("GRAPH_INIT_CONFIG_MAX_LATENCY_MS", "1600")
)  # Config mode aims for topology-like latency expectations.

INITIAL_GENERAL_MAX_HOPS: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_MAX_HOPS", "2")
)  # General mode defaults to conservative depth until intent becomes clearer.
INITIAL_GENERAL_MAX_NODES_PER_HOP: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_MAX_NODES_PER_HOP", "48")
)  # Neutral breadth for mixed or underspecified questions.
INITIAL_GENERAL_MAX_EDGES_PER_NODE: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_MAX_EDGES_PER_NODE", "6")
)  # Prevents early over-expansion in ambiguous queries.
INITIAL_GENERAL_GLOBAL_PATH_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_GLOBAL_PATH_BUDGET", "240")
)  # Lower initial budget keeps baseline behavior stable and explainable.
INITIAL_GENERAL_GLOBAL_NODE_BUDGET: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_GLOBAL_NODE_BUDGET", "240")
)  # Matches path budget to preserve proportional pruning.
INITIAL_GENERAL_MAX_LATENCY_MS: Final[int] = int(
    os.getenv("GRAPH_INIT_GENERAL_MAX_LATENCY_MS", "1600")
)  # Default latency target for unspecialized retrieval mode.

ESCALATION_CONFIDENCE_THRESHOLD: Final[float] = float(
    os.getenv("GRAPH_ESCALATION_CONFIDENCE_THRESHOLD", "0.62")
)  # If average path confidence drops below this, broaden search to recover recall.
ESCALATION_REASON_CONFIDENCE_THRESHOLD: Final[float] = float(
    os.getenv("GRAPH_ESCALATION_REASON_CONFIDENCE_THRESHOLD", "0.55")
)  # Lower confidence marker used to tag "low_path_score_floor" escalation reason.
STOP_CONFIDENCE_FLOOR: Final[float] = float(
    os.getenv("GRAPH_STOP_CONFIDENCE_FLOOR", "0.35")
)  # Stop exploring when frontier quality is too low to justify extra hops.
ESCALATED_MARGINAL_GAIN_FLOOR: Final[int] = int(
    os.getenv("GRAPH_ESCALATED_MARGINAL_GAIN_FLOOR", "1")
)  # After escalation, halt if each round adds <=1 new path (diminishing returns).

RANK_PROVENANCE_EVIDENCE_WEIGHT: Final[float] = float(
    os.getenv("GRAPH_RANK_PROVENANCE_EVIDENCE_WEIGHT", "0.08")
)  # Per-evidence-row bonus weight (capped later) to reward corroboration depth.
RANK_PROVENANCE_SOURCE_DIVERSITY_WEIGHT: Final[float] = float(
    os.getenv("GRAPH_RANK_PROVENANCE_SOURCE_DIVERSITY_WEIGHT", "0.06")
)  # Bonus for cross-source corroboration to reduce single-source bias.
RANK_PROVENANCE_LINE_BOUNDS_BONUS: Final[float] = float(
    os.getenv("GRAPH_RANK_PROVENANCE_LINE_BOUNDS_BONUS", "0.05")
)  # Small uplift when precise line references are available for citation quality.
RANK_PROVENANCE_MAX_MULTIPLIER: Final[float] = float(
    os.getenv("GRAPH_RANK_PROVENANCE_MAX_MULTIPLIER", "1.35")
)  # Prevents provenance bonuses from overwhelming semantic/path relevance.

RANK_QUERY_BASE_MULTIPLIER: Final[float] = float(
    os.getenv("GRAPH_RANK_QUERY_BASE_MULTIPLIER", "0.9")
)  # Base query relevance multiplier before coverage/exact-match boosts.
RANK_QUERY_COVERAGE_WEIGHT: Final[float] = float(
    os.getenv("GRAPH_RANK_QUERY_COVERAGE_WEIGHT", "0.35")
)  # Strength of term-coverage contribution to candidate relevance.
RANK_QUERY_EXACT_NODE_BONUS: Final[float] = float(
    os.getenv("GRAPH_RANK_QUERY_EXACT_NODE_BONUS", "0.35")
)  # Extra boost when candidate node name exactly matches a query term.
RANK_TOPOLOGY_MESSAGING_DOC_BONUS: Final[float] = float(
    os.getenv("GRAPH_RANK_TOPOLOGY_MESSAGING_DOC_BONUS", "0.25")
)  # Biases topology messaging queries toward config-heavy evidence docs.

RANK_DECAY_FIXED: Final[float] = float(os.getenv("GRAPH_RANK_DECAY_FIXED", "0.35"))  # Hop penalty decay used when policy mode is fixed depth.
RANK_DECAY_ADAPTIVE_ESCALATED: Final[float] = float(
    os.getenv("GRAPH_RANK_DECAY_ADAPTIVE_ESCALATED", "0.2")
)  # Softer decay after escalation so deeper rescued paths are not over-penalized.
RANK_DECAY_ADAPTIVE_DEFAULT: Final[float] = float(
    os.getenv("GRAPH_RANK_DECAY_ADAPTIVE_DEFAULT", "0.32")
)  # Default adaptive decay balancing shallow precision with deeper recall.
