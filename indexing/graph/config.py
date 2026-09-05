"""Graph indexing configuration defaults.

Tuning guide::
- Confidence constants affect evidence trust ranking during graph ingest.
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
CONTRACT_CSPROJ_METADATA_CONFIDENCE: Final[float] = float(
    os.getenv("GRAPH_CONTRACT_CSPROJ_METADATA_CONFIDENCE", "0.96")
)  # .csproj declarations are authoritative for framework/package metadata.
