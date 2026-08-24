"""Result writer for raw retrieval outputs and aggregated summaries."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.harness.schema import MetricRow, RetrievalErrorRow, RetrievalHit, RunMeta


class RunWriter:
    """Persists one retrieval evaluation run and derived summaries."""

    def __init__(self, results_dir: Path, run_meta: RunMeta) -> None:
        self.results_dir = results_dir
        self.run_meta = run_meta
        self.run_dir = self.results_dir / self.run_meta.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write(
        self,
        raw_hits: list[RetrievalHit],
        metric_rows: list[MetricRow],
        query_rows: list[dict[str, Any]],
        error_rows: list[RetrievalErrorRow],
    ) -> Path:
        self._write_run_meta()
        self._write_jsonl(self.run_dir / "retrieval_raw.jsonl", [hit.to_dict() for hit in raw_hits])
        self._write_csv(self.run_dir / "metrics_long.csv", [row.to_dict() for row in metric_rows])
        self._write_csv(self.run_dir / "summary_by_query.csv", query_rows)
        self._write_csv(self.run_dir / "errors.csv", [row.to_dict() for row in error_rows])
        self._write_csv(self.run_dir / "summary.csv", self._build_summary_rows(metric_rows))
        self._write_csv(
            self.run_dir / "summary_by_category.csv",
            self._build_summary_by_category_rows(metric_rows),
        )
        return self.run_dir

    def _write_run_meta(self) -> None:
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(self.run_meta.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _build_summary_rows(metric_rows: list[MetricRow]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in metric_rows:
            grouped[(row.strategy, row.k)][row.metric].append(row.value)

        rows: list[dict[str, Any]] = []
        for (strategy, k), metric_values in sorted(grouped.items()):
            row: dict[str, Any] = {
                "strategy": strategy,
                "k": k,
            }
            for metric_name, values in sorted(metric_values.items()):
                row[metric_name] = sum(values) / len(values) if values else 0.0
                row[f"{metric_name}_count"] = len(values)
            rows.append(row)
        return rows

    @staticmethod
    def _build_summary_by_category_rows(metric_rows: list[MetricRow]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in metric_rows:
            grouped[(row.strategy, row.category, row.k)][row.metric].append(row.value)

        rows: list[dict[str, Any]] = []
        for (strategy, category, k), metric_values in sorted(grouped.items()):
            row: dict[str, Any] = {
                "strategy": strategy,
                "category": category,
                "k": k,
            }
            for metric_name, values in sorted(metric_values.items()):
                row[metric_name] = sum(values) / len(values) if values else 0.0
                row[f"{metric_name}_count"] = len(values)
            rows.append(row)
        return rows
