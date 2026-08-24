"""Configuration defaults for retrieval baseline evaluation."""

from __future__ import annotations

import os
from pathlib import Path

_retrieval_host_port = os.getenv("RETRIEVAL_HOST_PORT", "18000")
DEFAULT_RETRIEVAL_URL = os.getenv(
	"RETRIEVAL_URL", f"http://localhost:{_retrieval_host_port}"
)
DEFAULT_STRATEGIES = ("keyword", "vector", "hybrid")
DEFAULT_K_VALUES = (5, 10)
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("EVAL_HTTP_TIMEOUT_SECONDS", "60"))

ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR / "evaluation"
DEFAULT_GOLDEN_JSON = EVALUATION_DIR / "golden_queries.json"
DEFAULT_DATASET_DIR = EVALUATION_DIR / "datasets" / "v1"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"
