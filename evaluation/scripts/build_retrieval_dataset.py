"""Build a frozen retrieval dataset from golden_queries.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.harness.config import DEFAULT_DATASET_DIR, DEFAULT_GOLDEN_JSON
from evaluation.harness.dataset import GoldenDatasetBuilder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_GOLDEN_JSON,
        help="Path to golden_queries.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory for frozen dataset outputs.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v1",
        help="Dataset version label.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset normalization and write frozen files."""

    args = parse_args()
    builder = GoldenDatasetBuilder(
        input_json=args.input_json,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
    )
    summary = builder.build()

    print(f"dataset_dir={args.output_dir}")
    print(f"query_count={summary.query_count}")
    print(f"qrel_count={summary.qrel_count}")
    print(f"dropped_unanswerable={summary.dropped_unanswerable}")
    print(f"dropped_missing_qrels={summary.dropped_missing_qrels}")


if __name__ == "__main__":
    main()
