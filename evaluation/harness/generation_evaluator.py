"""End-to-end generation evaluator using golden query answers."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluation.harness.generation_metrics import GenerationMetricsCalculator
from evaluation.harness.generation_schema import (
    GenerationQueryRecord,
    GenerationResultRow,
    GenerationRunMeta,
)


@dataclass(frozen=True)
class GenerationEvaluationRunResult:
    run_id: str
    query_count: int


class GenerationEvaluator:
    """Evaluates generation responses against gold_answer."""

    def __init__(
        self,
        *,
        golden_json_path: Path,
        generation_url: str,
        modes: tuple[str, ...],
        decomposition_policies: tuple[str, ...],
        timeout_seconds: int,
    ) -> None:
        self.golden_json_path = golden_json_path
        self.generation_url = generation_url.rstrip("/")
        self.modes = tuple(sorted(set(modes)))
        self.decomposition_policies = tuple(sorted(set(decomposition_policies)))
        self.timeout_seconds = timeout_seconds
        self.metrics = GenerationMetricsCalculator()

    def run(self, *, results_dir: Path, run_writer_cls: Any, limit: int | None = None) -> GenerationEvaluationRunResult:
        queries = self._load_queries(self.golden_json_path)
        if limit is not None and limit > 0:
            queries = queries[:limit]

        run_id = self._build_run_id()
        run_meta = GenerationRunMeta(
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            dataset_version="golden_queries_v1",
            dataset_hashes={"golden_queries.json": self._sha256_file(self.golden_json_path)},
            modes=self.modes,
            decomposition_policies=self.decomposition_policies,
            generation_url=self.generation_url,
            query_count=len(queries),
            config={"timeout_seconds": self.timeout_seconds},
            judge={"model": "heuristic-v1", "prompt_version": "heuristic-v1"},
        )

        rows: list[dict[str, Any]] = []
        for query in queries:
            rows.extend(self._evaluate_query(run_id=run_id, query=query))

        run_writer = run_writer_cls(results_dir=results_dir, run_meta=run_meta)
        run_writer.write(results_rows=rows)
        return GenerationEvaluationRunResult(run_id=run_id, query_count=len(queries))

    def _evaluate_query(self, *, run_id: str, query: GenerationQueryRecord) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mode in self.modes:
            for policy in self.decomposition_policies:
                try:
                    payload = self._call_generation(query=query, mode=mode, decomposition_policy=policy)
                    row = self._build_result_row(
                        run_id=run_id,
                        query=query,
                        mode=mode,
                        decomposition_policy=policy,
                        payload=payload,
                        error="",
                    )
                except RuntimeError as exc:
                    row = self._build_result_row(
                        run_id=run_id,
                        query=query,
                        mode=mode,
                        decomposition_policy=policy,
                        payload={},
                        error=str(exc),
                    )
                rows.append(row.to_dict())
        return rows

    def _call_generation(self, *, query: GenerationQueryRecord, mode: str, decomposition_policy: str) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query.query_text, "top_k": 10, "mode": mode}
        if decomposition_policy != "auto":
            body["decompose_override"] = decomposition_policy

        req = Request(
            url=f"{self.generation_url}/query",
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} from {self.generation_url}/query") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach generation URL {self.generation_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Timed out waiting for generation response after {self.timeout_seconds}s") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Generation response was not a JSON object.")
        return payload

    def _build_result_row(
        self,
        *,
        run_id: str,
        query: GenerationQueryRecord,
        mode: str,
        decomposition_policy: str,
        payload: dict[str, Any],
        error: str,
    ) -> GenerationResultRow:
        answer_text = str(payload.get("answer", ""))
        citations = payload.get("citations", [])
        citation_rows = citations if isinstance(citations, list) else []
        cited_services, cited_files = self._extract_citation_sets(citation_rows)
        gold_services = set(query.gold_services)
        gold_files = {self._normalize_path(item) for item in query.gold_source_files}

        overlap_services = self.metrics.overlap_ratio(cited_services, gold_services)
        overlap_files = self.metrics.overlap_ratio(cited_files, gold_files)
        citation_precision = self.metrics.citation_precision(
            cited_files=cited_files,
            cited_services=cited_services,
            gold_files=gold_files,
            gold_services=gold_services,
        )
        citation_recall_proxy = overlap_files if gold_files else overlap_services

        score = self.metrics.score_answer(
            generated_answer=answer_text,
            gold_answer=query.gold_answer,
            answerable=query.answerable,
            citations_count=len(citation_rows),
        )
        return GenerationResultRow(
            run_id=run_id,
            query_id=query.query_id,
            query_text=query.query_text,
            category=query.category,
            difficulty=query.difficulty,
            answerable=query.answerable,
            mode=mode,
            decomposition_policy=decomposition_policy,
            metadata_mode=str(payload.get("metadata_mode", "")),
            detected_services=self._stringify(payload.get("detected_services")),
            source_searched=str(payload.get("source_searched", "")),
            generated_answer=answer_text,
            gold_answer=query.gold_answer,
            citations_count=len(citation_rows),
            citation_sources=";".join(sorted(cited_services)),
            latency_ms=self._as_float(payload.get("latency_ms")),
            retrieval_latency_ms=self._as_float(payload.get("retrieval_latency_ms")),
            generation_latency_ms=self._as_float(payload.get("generation_latency_ms")),
            decomposition_used=bool(payload.get("decomposition_used", False)),
            decomposition_reason=str(payload.get("decomposition_reason", "")),
            decomposition_branches=self._stringify(payload.get("decomposition_branches")),
            branch_result_counts=json.dumps(payload.get("branch_result_counts", {}), sort_keys=True),
            correctness_score=score.correctness_score,
            completeness_score=score.completeness_score,
            abstention_quality_score=score.abstention_quality_score,
            hallucination_flag=score.hallucination_flag,
            hallucination_notes=score.hallucination_notes,
            faithfulness_score=score.faithfulness_score,
            citation_precision=citation_precision,
            citation_recall_proxy=citation_recall_proxy,
            gold_service_overlap=overlap_services,
            gold_file_overlap=overlap_files,
            judge_model="heuristic-v1",
            judge_prompt_version="heuristic-v1",
            judge_rationale=score.judge_rationale,
            score_confidence=score.score_confidence,
            error=error,
        )

    @staticmethod
    def _extract_citation_sets(citations: list[Any]) -> tuple[set[str], set[str]]:
        services: set[str] = set()
        files: set[str] = set()
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            metadata = citation.get("metadata")
            if isinstance(metadata, dict):
                source = str(metadata.get("source", "")).strip()
                if source:
                    services.add(source)
                file_path = str(metadata.get("file_path") or metadata.get("file") or metadata.get("path") or "").strip()
                if file_path:
                    files.add(GenerationEvaluator._normalize_path(file_path))
            title = str(citation.get("title", "")).strip()
            if title:
                files.add(GenerationEvaluator._normalize_path(title))
        return services, files

    @staticmethod
    def _normalize_path(value: str) -> str:
        return value.strip().lstrip("./")

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, list):
            return ";".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        return str(value or "")

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"generation_{now}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _load_queries(path: Path) -> list[GenerationQueryRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("queries", []) if isinstance(payload, dict) else []
        records: list[GenerationQueryRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            query_id = str(row.get("master_query_id", "")).strip()
            query_text = str(row.get("query", "")).strip()
            gold_answer = str(row.get("gold_answer", "")).strip()
            if not query_id or not query_text or not gold_answer:
                continue
            records.append(
                GenerationQueryRecord(
                    query_id=query_id,
                    query_text=query_text,
                    gold_answer=gold_answer,
                    category=str(row.get("category", "")),
                    difficulty=int(row.get("difficulty", 0) or 0),
                    answerable=bool(row.get("answerable", True)),
                    wording_type=str(row.get("wording_type", "")),
                    gold_services=tuple(sorted({str(item).strip() for item in row.get("gold_services", []) if str(item).strip()})),
                    gold_source_files=tuple(
                        sorted(
                            {
                                GenerationEvaluator._normalize_path(str(item))
                                for item in row.get("gold_source_files", [])
                                if str(item).strip()
                            }
                        )
                    ),
                )
            )
        return records
