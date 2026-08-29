"""Configuration defaults for retrieval baseline evaluation."""

from __future__ import annotations

import os
from pathlib import Path

_retrieval_host_port = os.getenv("RETRIEVAL_HOST_PORT", "18000")
DEFAULT_RETRIEVAL_URL = os.getenv(
	"RETRIEVAL_URL", f"http://localhost:{_retrieval_host_port}"
)
_generation_host_port = os.getenv("GENERATION_HOST_PORT", "18002")
DEFAULT_GENERATION_URL = os.getenv(
    "GENERATION_URL", f"http://localhost:{_generation_host_port}"
)
DEFAULT_STRATEGIES = (
    "graph-service-aware",
    "hybrid",
    "hybrid-service-aware",
    "keyword-service-aware",
    "vector-service-aware",
)
DEFAULT_K_VALUES = (10, 15)
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("EVAL_HTTP_TIMEOUT_SECONDS", "60"))
DEFAULT_GENERATION_TIMEOUT_SECONDS = int(os.getenv("GEN_EVAL_HTTP_TIMEOUT_SECONDS", "90"))
DEFAULT_GENERATION_MODES = ("hybrid", "graph")
DEFAULT_DECOMPOSITION_POLICIES = ("auto",)

ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR / "evaluation"
DEFAULT_GOLDEN_JSON = EVALUATION_DIR / "golden_queries.json"
DEFAULT_DATASET_DIR = EVALUATION_DIR / "datasets" / "v1"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_SERVICE_CATALOGUE_PATH = ROOT_DIR / "service_acronyms.json"
