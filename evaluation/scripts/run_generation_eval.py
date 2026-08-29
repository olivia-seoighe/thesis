"""Run end-to-end generation evaluation against golden_answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.harness.config import (
    DEFAULT_DECOMPOSITION_POLICIES,
    DEFAULT_GENERATION_MODES,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_GENERATION_URL,
    DEFAULT_GOLDEN_JSON,
    DEFAULT_RESULTS_DIR,
)
from evaluation.harness.generation_evaluator import GenerationEvaluator
from evaluation.harness.generation_results import GenerationRunWriter


def _parse_csv(raw: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    return values or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-json",
        type=Path,
        default=DEFAULT_GOLDEN_JSON,
        help="Path to golden_queries.json containing query + gold_answer rows.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--generation-url",
        type=str,
        default=DEFAULT_GENERATION_URL,
        help="Base URL for generation API.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_GENERATION_MODES),
        help="Comma-separated generation retrieval modes (e.g. hybrid,graph).",
    )
    parser.add_argument(
        "--decomposition-policies",
        type=str,
        default=",".join(DEFAULT_DECOMPOSITION_POLICIES),
        help="Comma-separated decomposition policies (auto,on,off).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        help="HTTP timeout per generation request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional query limit for smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = GenerationEvaluator(
        golden_json_path=args.golden_json,
        generation_url=args.generation_url,
        modes=_parse_csv(args.modes, DEFAULT_GENERATION_MODES),
        decomposition_policies=_parse_csv(args.decomposition_policies, DEFAULT_DECOMPOSITION_POLICIES),
        timeout_seconds=args.timeout_seconds,
    )
    result = evaluator.run(
        results_dir=args.results_dir,
        run_writer_cls=GenerationRunWriter,
        limit=args.limit,
    )
    print(f"run_id={result.run_id}")
    print(f"query_count={result.query_count}")
    print(f"results_dir={(args.results_dir / result.run_id).as_posix()}")


if __name__ == "__main__":
    main()
