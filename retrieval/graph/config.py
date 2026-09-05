"""Configuration defaults for graph retrieval behavior.

Tuning guide:
- Budget/cap constants trade retrieval recall vs latency.
- Threshold constants control seed and structured-route confidence.
"""

from __future__ import annotations

import os
from typing import Final

GRAPH_NAME: Final[str] = os.getenv("GRAPH_NAME", "enterprise_graph")  # Must match ingestion/indexing graph name so Cypher targets the same graph.

SEED_LOOKUP_LIMIT: Final[int] = int(os.getenv("GRAPH_SEED_LOOKUP_LIMIT", "15"))  # Caps seed fan-out so ambiguous mentions do not explode traversal.
TOPOLOGY_QUERY_LIMIT: Final[int] = int(os.getenv("GRAPH_TOPOLOGY_QUERY_LIMIT", "500"))  # Keeps topology endpoint bounded for latency and payload size.

MAX_SEED_MATCHES: Final[int] = int(os.getenv("GRAPH_MAX_SEED_MATCHES", "25"))  # Round-robin cap across mention groups before traversal planning.
EMBEDDING_SEED_TOP_K_PER_LABEL: Final[int] = int(
    os.getenv("GRAPH_EMBEDDING_SEED_TOP_K_PER_LABEL", "10")
)  # Embedding candidates per label.
EMBEDDING_SEED_MIN_SIMILARITY: Final[float] = float(
    os.getenv("GRAPH_EMBEDDING_SEED_MIN_SIMILARITY", "0.11")
)  # Cosine-similarity floor for embedding seed candidates.
EMBEDDING_SEED_TIMEOUT_SECONDS: Final[float] = float(
    os.getenv("GRAPH_EMBEDDING_SEED_TIMEOUT_SECONDS", "2.5")
)  # Timeout for embedding seed lookup.

STRUCTURED_QUERY_MIN_SIMILARITY: Final[float] = float(
    os.getenv("GRAPH_STRUCTURED_QUERY_MIN_SIMILARITY", "0.51")
)  # Cosine-similarity floor for structured-query routing.

RANK_DECAY: Final[float] = float(os.getenv("GRAPH_RANK_DECAY", "0.35"))  # Hop penalty decay for graph paths.
