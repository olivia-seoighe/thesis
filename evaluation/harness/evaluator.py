"""Retrieval evaluator for baseline and service-aware strategy variants."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.harness.metrics import MetricsCalculator
from evaluation.harness.schema import EvaluationResultRow, RunMeta
from evaluation.harness.strategies import StrategyRunner
from retrieval.strategies.metadata_aware import (
    SERVICE_AWARE_SUFFIX,
    ServiceAwarePlanner,
    build_service_planner,
    resolve_strategy_decision,
    run_service_aware_strategy,
)


@dataclass(frozen=True)
class EvaluationRunResult:
    """Paths and summary values from a retrieval run."""

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
        service_catalogue_path: Path | None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.retrieval_url = retrieval_url
        self.strategies = strategies
        self.k_values = tuple(sorted(set(k_values)))
        self.timeout_seconds = timeout_seconds
        self.service_catalogue_path = service_catalogue_path
        self.metrics = MetricsCalculator()
        self.strategy_runner = StrategyRunner(retrieval_url, timeout_seconds=timeout_seconds)

    def run(self, results_dir: Path, run_writer_cls: Any, limit: int | None = None) -> EvaluationRunResult:
        queries = self._load_jsonl(self.dataset_dir / "queries_v1.jsonl")
        qrels = self._load_jsonl(self.dataset_dir / "qrels_v1.jsonl")
        dataset_meta = self._load_dataset_meta(self.dataset_dir / "dataset_meta.json")
        if limit is not None and limit > 0:
            queries = queries[:limit]

        qrel_scheme = str(dataset_meta.get("qrel_scheme", "binary")).strip().lower()
        if qrel_scheme not in {"binary", "graded"}:
            qrel_scheme = "binary"
        qrel_grades_by_query = self._build_qrel_grades_by_query(qrels)
        qrels_by_query = self._build_relevant_doc_ids_by_query(
            qrel_grades_by_query=qrel_grades_by_query,
            qrel_scheme=qrel_scheme,
        )
        run_id = self._build_run_id()
        has_service_aware_variant = any(strategy.endswith(SERVICE_AWARE_SUFFIX) for strategy in self.strategies)
        planner = (
            build_service_planner(
                list_sources=self.strategy_runner.list_sources,
                service_catalogue_path=self.service_catalogue_path,
            )
            if has_service_aware_variant
            else ServiceAwarePlanner(())
        )

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
                "max_k": max(self.k_values),
                "service_aware_enabled": has_service_aware_variant,
                "metadata_boost_mode": "rrf",
                "qrel_scheme": qrel_scheme,
                "service_catalogue_path": str(self.service_catalogue_path) if self.service_catalogue_path else "retrieval:/sources",
            },
        )

        results_rows: list[dict[str, Any]] = []
        for query in queries:
            results_rows.extend(
                self._evaluate_query(
                    run_id=run_id,
                    query=query,
                    qrels_by_query=qrels_by_query,
                    qrel_grades_by_query=qrel_grades_by_query,
                    qrel_scheme=qrel_scheme,
                    planner=planner,
                )
            )

        run_writer = run_writer_cls(results_dir=results_dir, run_meta=run_meta)
        run_writer.write(results_rows=results_rows)
        return EvaluationRunResult(run_id=run_id, query_count=len(queries), qrel_count=len(qrels))

    def _evaluate_query(
        self,
        *,
        run_id: str,
        query: dict[str, Any],
        qrels_by_query: dict[str, set[str]],
        qrel_grades_by_query: dict[str, dict[str, int]],
        qrel_scheme: str,
        planner: ServiceAwarePlanner,
    ) -> list[dict[str, Any]]:
        query_id = str(query["query_id"])
        query_text = str(query["query_text"])
        category = str(query["category"])
        difficulty = self._as_int(query.get("difficulty"), default=0)
        relevant_doc_ids = qrels_by_query.get(query_id, set())
        relevance_by_doc_id = qrel_grades_by_query.get(query_id, {})
        decision = planner.plan(query_text)

        rows: list[dict[str, Any]] = []
        for strategy in self.strategies:
            base_strategy, execution_decision = resolve_strategy_decision(
                strategy=strategy,
                planned_decision=decision,
            )
            outcome = run_service_aware_strategy(
                search=self.strategy_runner.search,
                base_strategy=base_strategy,
                query_text=query_text,
                decision=execution_decision,
                top_k=max(self.k_values),
            )
            unique_doc_ids = self.metrics.unique_doc_ids_in_order([chunk.document_id for chunk in outcome.chunks])
            for k in self.k_values:
                recall = self.metrics.recall_at_k(unique_doc_ids, relevant_doc_ids, k)
                precision = self.metrics.precision_at_k(unique_doc_ids, relevant_doc_ids, k)
                f1 = self.metrics.f1_score(recall, precision)
                hit_count = self.metrics.hit_count_at_k(unique_doc_ids, relevant_doc_ids, k)
                mrr = self.metrics.mrr_at_k(unique_doc_ids, relevant_doc_ids, k)
                if qrel_scheme == "graded":
                    ndcg = self.metrics.ndcg_at_k_graded(unique_doc_ids, relevance_by_doc_id, k)
                else:
                    ndcg = self.metrics.ndcg_at_k(unique_doc_ids, relevant_doc_ids, k)
                rows.append(
                    EvaluationResultRow(
                        run_id=run_id,
                        query_id=query_id,
                        query_text=query_text,
                        category=category,
                        difficulty=difficulty,
                        strategy=strategy,
                        k=k,
                        metadata_mode=outcome.metadata_mode,
                        detected_services=";".join(outcome.detected_services),
                        recall=recall,
                        precision=precision,
                        f1=f1,
                        mrr=mrr,
                        ndcg=ndcg,
                        relevant_count=len(relevant_doc_ids),
                        retrieved_count=min(k, len(unique_doc_ids)),
                        hit_count=hit_count,
                        error=outcome.strategy_error,
                    ).to_dict()
                )
        return rows

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
    def _build_qrel_grades_by_query(qrels: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        qrels_by_query: dict[str, dict[str, int]] = {}
        for qrel in qrels:
            query_id = str(qrel.get("query_id", "")).strip()
            doc_id = str(qrel.get("doc_id", "")).strip()
            if not query_id or not doc_id:
                continue
            relevance = RetrievalBaselineEvaluator._as_int(qrel.get("relevance"), default=1)
            query_grades = qrels_by_query.setdefault(query_id, {})
            current = query_grades.get(doc_id)
            if current is None or relevance > current:
                query_grades[doc_id] = relevance
        return qrels_by_query

    @staticmethod
    def _build_relevant_doc_ids_by_query(
        *,
        qrel_grades_by_query: dict[str, dict[str, int]],
        qrel_scheme: str,
    ) -> dict[str, set[str]]:
        min_relevance = 2 if qrel_scheme == "graded" else 1
        relevant_doc_ids_by_query: dict[str, set[str]] = {}
        for query_id, doc_grades in qrel_grades_by_query.items():
            relevant_doc_ids_by_query[query_id] = {
                doc_id
                for doc_id, relevance in doc_grades.items()
                if relevance >= min_relevance
            }
        return relevant_doc_ids_by_query

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        return f"retrieval_{now}_{suffix}"
