"""Run keyword/vector/hybrid retrieval baseline against frozen dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.harness.config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_K_VALUES,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RETRIEVAL_URL,
    DEFAULT_SERVICE_CATALOGUE_PATH,
    DEFAULT_STRATEGIES,
    DEFAULT_TIMEOUT_SECONDS,
)
from evaluation.harness.evaluator import RetrievalBaselineEvaluator
from evaluation.harness.results import RunWriter


def parse_k_values(raw_value: str) -> tuple[int, ...]:
    """Parse comma-separated k values into a sorted tuple."""

    values: set[int] = set()
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        values.add(int(token))
    if not values:
        return DEFAULT_K_VALUES
    return tuple(sorted(values))


def parse_strategies(raw_value: str) -> tuple[str, ...]:
    """Parse comma-separated strategy names into a tuple."""

    values: list[str] = []
    for part in raw_value.split(","):
        token = part.strip().lower()
        if token:
            values.append(token)
    return tuple(values) if values else DEFAULT_STRATEGIES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for retrieval baseline runs."""

    default_service_catalogue = (
        DEFAULT_SERVICE_CATALOGUE_PATH if DEFAULT_SERVICE_CATALOGUE_PATH.exists() else None
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Frozen dataset directory containing queries_v1.jsonl and qrels_v1.jsonl.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for run artifacts.",
    )
    parser.add_argument(
        "--retrieval-url",
        type=str,
        default=DEFAULT_RETRIEVAL_URL,
        help="Base URL for retrieval API.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help=(
            "Comma-separated strategies. Default subset: "
            "graph-service-aware,hybrid,hybrid-service-aware,keyword-service-aware,vector-service-aware. "
            "Keyword variants: keyword (default fts), keyword-fts, keyword-bm25. "
            "Hybrid variants: hybrid (default fts), hybrid-fts, hybrid-bm25. "
            "Graph variants: graph-adaptive, graph-fixed. "
            "Service-aware variant: append -service-aware (e.g. vector-service-aware)."
        ),
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated k values.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional query limit for smoke runs.",
    )
    parser.add_argument(
        "--service-catalogue",
        type=Path,
        default=default_service_catalogue,
        help=(
            "Optional JSON file with service catalogue (source + short/long forms). "
            "Defaults to service_acronyms.json at repo root when present; "
            "otherwise evaluator uses retrieval /sources as the catalogue."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute retrieval baseline and persist run outputs."""

    args = parse_args()
    k_values = parse_k_values(args.k_values)
    strategies = parse_strategies(args.strategies)

    evaluator = RetrievalBaselineEvaluator(
        dataset_dir=args.dataset_dir,
        retrieval_url=args.retrieval_url,
        strategies=strategies,
        k_values=k_values,
        timeout_seconds=args.timeout_seconds,
        service_catalogue_path=args.service_catalogue,
    )

    result = evaluator.run(
        results_dir=args.results_dir,
        run_writer_cls=RunWriter,
        limit=args.limit,
    )

    print(f"run_id={result.run_id}")
    print(f"query_count={result.query_count}")
    print(f"qrel_count={result.qrel_count}")
    print(f"results_dir={(args.results_dir / result.run_id).as_posix()}")


if __name__ == "__main__":
    main()
