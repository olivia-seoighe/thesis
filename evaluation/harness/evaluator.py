"""Retrieval baseline evaluator for keyword, vector, and hybrid strategies."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.harness.metrics import MetricsCalculator
from evaluation.harness.schema import MetricRow, RetrievalErrorRow, RetrievalHit, RunMeta
from evaluation.harness.strategies import StrategyRunner


@dataclass(frozen=True)
class EvaluationRunResult:
    """Paths and summary values from a retrieval baseline run."""

    run_id: str
    query_count: int
    qrel_count: int


class RetrievalBaselineEvaluator:
    """Evaluates retrieval strategies against frozen queries and qrels."""

    def __init__(
        self,
        dataset_dir: Path,
        retrieval_url: str,
        strategies: tuple[str, ...],
        k_values: tuple[int, ...],
        timeout_seconds: int,
        use_source_filters: bool,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.retrieval_url = retrieval_url
        self.strategies = strategies
        self.k_values = tuple(sorted(set(k_values)))
        self.timeout_seconds = timeout_seconds
        self.use_source_filters = use_source_filters
        self.metrics = MetricsCalculator()
        self.strategy_runner = StrategyRunner(retrieval_url, timeout_seconds=timeout_seconds)

    def run(self, results_dir: Path, run_writer_cls: Any, limit: int | None = None) -> EvaluationRunResult:
        queries = self._load_jsonl(self.dataset_dir / "queries_v1.jsonl")
        qrels = self._load_jsonl(self.dataset_dir / "qrels_v1.jsonl")
        dataset_meta = self._load_dataset_meta(self.dataset_dir / "dataset_meta.json")

        if limit is not None and limit > 0:
            queries = queries[:limit]

        qrels_by_query = self._build_qrels_by_query(qrels)
        run_id = self._build_run_id()

        run_meta = RunMeta(
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            dataset_version=str(dataset_meta.get("dataset_version", "v1")),
            dataset_hashes=dict(dataset_meta.get("hashes", {})),
            strategies=self.strategies,
            k_values=self.k_values,
            retrieval_url=self.retrieval_url,
            query_count=len(queries),
            qrel_count=len(qrels),
            config={
                "timeout_seconds": self.timeout_seconds,
                "use_source_filters": self.use_source_filters,
                "max_k": max(self.k_values),
            },
        )

        raw_hits: list[RetrievalHit] = []
        error_rows: list[RetrievalErrorRow] = []
        metric_rows: list[MetricRow] = []
        query_rows: list[dict[str, Any]] = []

        for query in queries:
            query_id = str(query["query_id"])
            query_text = str(query["query_text"])
            category = str(query["category"])
            services = tuple(query.get("services", []))
            relevant_doc_ids = qrels_by_query.get(query_id, set())

            for strategy in self.strategies:
                chunks = []
                strategy_error = ""
                try:
                    chunks = self.strategy_runner.search(
                        strategy=strategy,
                        query_text=query_text,
                        top_k=max(self.k_values),
                        sources=services if self.use_source_filters else (),
                    )
                except RuntimeError as exc:
                    strategy_error = str(exc)
                    for k in self.k_values:
                        error_rows.append(
                            RetrievalErrorRow(
                                run_id=run_id,
                                query_id=query_id,
                                category=category,
                                strategy=strategy,
                                k=k,
                                error=strategy_error,
                            )
                        )

                hits: list[RetrievalHit] = []
                for rank, chunk in enumerate(chunks, start=1):
                    hits.append(
                        RetrievalHit(
                            query_id=query_id,
                            category=category,
                            strategy=strategy,
                            rank=rank,
                            doc_id=chunk.document_id,
                            chunk_id=chunk.chunk_id,
                            score=chunk.score,
                        )
                    )
                raw_hits.extend(hits)

                unique_doc_ids = self.metrics.unique_doc_ids_in_order([hit.doc_id for hit in hits])

                for k in self.k_values:
                    recall = self.metrics.recall_at_k(unique_doc_ids, relevant_doc_ids, k)
                    precision = self.metrics.precision_at_k(unique_doc_ids, relevant_doc_ids, k)
                    f1 = self.metrics.f1_score(recall, precision)
                    hit_count = self.metrics.hit_count_at_k(unique_doc_ids, relevant_doc_ids, k)
                    retrieved_count = min(k, len(unique_doc_ids))
                    relevant_count = len(relevant_doc_ids)

                    metric_rows.append(
                        MetricRow(
                            run_id=run_id,
                            query_id=query_id,
                            category=category,
                            strategy=strategy,
                            k=k,
                            metric="recall",
                            value=recall,
                            relevant_count=relevant_count,
                            retrieved_count=retrieved_count,
                            hit_count=hit_count,
                        )
                    )
                    metric_rows.append(
                        MetricRow(
                            run_id=run_id,
                            query_id=query_id,
                            category=category,
                            strategy=strategy,
                            k=k,
                            metric="precision",
                            value=precision,
                            relevant_count=relevant_count,
                            retrieved_count=retrieved_count,
                            hit_count=hit_count,
                        )
                    )
                    metric_rows.append(
                        MetricRow(
                            run_id=run_id,
                            query_id=query_id,
                            category=category,
                            strategy=strategy,
                            k=k,
                            metric="f1",
                            value=f1,
                            relevant_count=relevant_count,
                            retrieved_count=retrieved_count,
                            hit_count=hit_count,
                        )
                    )

                    query_rows.append(
                        {
                            "run_id": run_id,
                            "query_id": query_id,
                            "query_text": query_text,
                            "category": category,
                            "strategy": strategy,
                            "k": k,
                            "recall": recall,
                            "precision": precision,
                            "f1": f1,
                            "relevant_count": relevant_count,
                            "retrieved_count": retrieved_count,
                            "hit_count": hit_count,
                            "error": strategy_error,
                        }
                    )

        run_writer = run_writer_cls(results_dir=results_dir, run_meta=run_meta)
        run_writer.write(
            raw_hits=raw_hits,
            metric_rows=metric_rows,
            query_rows=query_rows,
            error_rows=error_rows,
        )

        return EvaluationRunResult(
            run_id=run_id,
            query_count=len(queries),
            qrel_count=len(qrels),
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    @staticmethod
    def _load_dataset_meta(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _build_qrels_by_query(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
        qrels_by_query: dict[str, set[str]] = {}
        for qrel in qrels:
            query_id = str(qrel.get("query_id", "")).strip()
            doc_id = str(qrel.get("doc_id", "")).strip()
            if not query_id or not doc_id:
                continue
            qrels_by_query.setdefault(query_id, set()).add(doc_id)
        return qrels_by_query

    @staticmethod
    def _build_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        return f"retrieval_{now}_{suffix}"
