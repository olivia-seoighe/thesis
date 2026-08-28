"""Graph indexing configuration defaults.

Tuning guide::
- Confidence constants affect evidence trust ranking during graph ingest.
- Source priority constants resolve conflicts when multiple sources describe the same entity.
"""

from __future__ import annotations

import os
from typing import Final

GRAPH_NAME: Final[str] = os.getenv("GRAPH_NAME", "enterprise_graph")  # Keeps runtime graph name aligned with schema bootstrap default.

AST_OWNERSHIP_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_AST_OWNERSHIP_CONFIDENCE", "0.97")
)  # High trust: ownership edges come from deterministic code symbols.
AST_RELATION_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_AST_RELATION_CONFIDENCE", "0.95")
)  # Slightly lower: handler/saga relation inference is deterministic but pattern-based.
AST_FEATURE_FLAG_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_AST_FEATURE_FLAG_CONFIDENCE", "0.93")
)  # Lower than ownership: flag detection relies on naming and usage patterns.
CONTRACT_TOPIC_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_TOPIC_CONFIDENCE", "0.98")
)  # Highest trust: topics usually come from explicit contract/config declarations.
CONTRACT_API_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_API_CONFIDENCE", "0.95")
)  # High trust: API names are explicit but can include normalization ambiguity.
CONTRACT_FLAG_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_FLAG_CONFIDENCE", "0.90")
)  # Most conservative: config flags can include environment/noise tokens.
CONTRACT_EXPOSES_API_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_EXPOSES_API_CONFIDENCE", "0.92")
)  # Ingress exposure is strong signal but may include generic host/path templates.
CONTRACT_TABLE_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_TABLE_CONFIDENCE", "0.96")
)  # Schema-state tables are authoritative DDL-level entities for table ownership.

SOURCE_PRIORITY_ASYNCAPI: Final[int] = int(os.getenv("GRAPH_SOURCE_PRIORITY_ASYNCAPI", "7"))  # Rank 7: canonical source for message-channel topology.
SOURCE_PRIORITY_AST: Final[int] = int(os.getenv("GRAPH_SOURCE_PRIORITY_AST", "6"))  # Rank 6: strong local truth for code-level behavior.
SOURCE_PRIORITY_CONFIGMAP: Final[int] = int(os.getenv("GRAPH_SOURCE_PRIORITY_CONFIGMAP", "5"))  # Rank 5: deployment config with service-specific wiring.
SOURCE_PRIORITY_APPSETTINGS_PROD: Final[int] = int(
    os.getenv("GRAPH_SOURCE_PRIORITY_APPSETTINGS_PROD", "4")
)  # Rank 4: production app settings, useful but environment-specific.
SOURCE_PRIORITY_APPSETTINGS_BASE: Final[int] = int(
    os.getenv("GRAPH_SOURCE_PRIORITY_APPSETTINGS_BASE", "3")
)  # Rank 3: baseline app settings, broader but less environment-grounded.
SOURCE_PRIORITY_INGRESS: Final[int] = int(os.getenv("GRAPH_SOURCE_PRIORITY_INGRESS", "2"))  # Rank 2: external exposure view, not full internal topology.
SOURCE_PRIORITY_DEFAULT: Final[int] = int(os.getenv("GRAPH_SOURCE_PRIORITY_DEFAULT", "1"))  # Rank 1: fallback for unknown/least-specific source kinds.
