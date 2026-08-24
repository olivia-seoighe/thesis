"""Builds a frozen retrieval dataset from golden_queries.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.harness.schema import QrelRecord, QueryRecord


@dataclass(frozen=True)
class DatasetBuildSummary:
    """Summary values emitted after dataset build."""

    query_count: int
    qrel_count: int
    dropped_unanswerable: int
    dropped_missing_qrels: int


class GoldenDatasetBuilder:
    """Normalizes golden query JSON into frozen query and qrels files."""

    def __init__(self, input_json: Path, output_dir: Path, dataset_version: str) -> None:
        self.input_json = input_json
        self.output_dir = output_dir
        self.dataset_version = dataset_version

    def build(self) -> DatasetBuildSummary:
        payload = json.loads(self.input_json.read_text(encoding="utf-8"))
        raw_queries = payload.get("queries", [])

        query_records: list[QueryRecord] = []
        qrel_records: list[QrelRecord] = []
        dropped_unanswerable = 0
        dropped_missing_qrels = 0

        for raw_query in raw_queries:
            if not isinstance(raw_query, dict):
                continue
            if not self._is_answerable(raw_query.get("answerable")):
                dropped_unanswerable += 1
                continue

            query_id = str(raw_query.get("master_query_id", "")).strip()
            query_text = str(raw_query.get("query", "")).strip()
            category = str(raw_query.get("category", "Unknown")).strip() or "Unknown"
            difficulty = self._as_int(raw_query.get("difficulty"), default=0)
            wording_type = str(raw_query.get("wording_type", "Unknown")).strip() or "Unknown"
            services = tuple(self._parse_services(raw_query.get("gold_services")))

            source_pairs = self._parse_gold_source_files(raw_query, services)
            source_pairs = self._canonicalize_source_pairs(source_pairs)

            if not query_id or not query_text or not source_pairs:
                dropped_missing_qrels += 1
                continue

            query_records.append(
                QueryRecord(
                    query_id=query_id,
                    query_text=query_text,
                    category=category,
                    difficulty=difficulty,
                    wording_type=wording_type,
                    services=services,
                )
            )

            for service, file_path in source_pairs:
                doc_id = self._build_doc_id(service, file_path)
                qrel_records.append(
                    QrelRecord(
                        query_id=query_id,
                        doc_id=doc_id,
                        service=service,
                        file_path=file_path,
                        relevance=1,
                    )
                )

        qrel_records = self._dedupe_qrels(qrel_records)
        query_records = sorted(query_records, key=lambda q: q.query_id)
        qrel_records = sorted(qrel_records, key=lambda q: (q.query_id, q.doc_id))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        queries_path = self.output_dir / "queries_v1.jsonl"
        qrels_path = self.output_dir / "qrels_v1.jsonl"

        self._write_jsonl(queries_path, [q.to_dict() for q in query_records])
        self._write_jsonl(qrels_path, [q.to_dict() for q in qrel_records])

        dataset_meta = {
            "dataset_version": self.dataset_version,
            "source_file": str(self.input_json),
            "query_count": len(query_records),
            "qrel_count": len(qrel_records),
            "dropped_unanswerable": dropped_unanswerable,
            "dropped_missing_qrels": dropped_missing_qrels,
            "hashes": {
                "queries_v1.jsonl": self._sha256_file(queries_path),
                "qrels_v1.jsonl": self._sha256_file(qrels_path),
            },
        }
        (self.output_dir / "dataset_meta.json").write_text(
            json.dumps(dataset_meta, indent=2) + "\n",
            encoding="utf-8",
        )

        return DatasetBuildSummary(
            query_count=len(query_records),
            qrel_count=len(qrel_records),
            dropped_unanswerable=dropped_unanswerable,
            dropped_missing_qrels=dropped_missing_qrels,
        )

    @staticmethod
    def _is_answerable(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().upper() == "TRUE"

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _parse_services(raw_value: Any) -> list[str]:
        if isinstance(raw_value, list):
            services = [str(s).strip() for s in raw_value if str(s).strip()]
            return sorted(set(services))
        if isinstance(raw_value, str):
            services = [s.strip() for s in raw_value.split(";") if s.strip()]
            return sorted(set(services))
        return []

    def _parse_gold_source_files(
        self,
        raw_query: dict[str, Any],
        services: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        raw_sources = raw_query.get("gold_source_files")
        pairs: list[tuple[str, str]] = []

        if isinstance(raw_sources, list):
            for item in raw_sources:
                if isinstance(item, dict):
                    service = str(item.get("service", "")).strip()
                    file_path = str(item.get("file", "")).strip()
                    if service and file_path:
                        pairs.append((service, file_path))
                elif isinstance(item, str):
                    pairs.extend(self._parse_source_string(item, services))
        elif isinstance(raw_sources, str):
            pairs.extend(self._parse_source_string(raw_sources, services))

        return pairs

    def _parse_source_string(
        self,
        raw_value: str,
        services: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        chunks: list[str] = []
        if "\n" in raw_value:
            chunks = [line.strip() for line in raw_value.splitlines() if line.strip()]
        else:
            chunks = [part.strip() for part in raw_value.split(";") if part.strip()]

        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            if "::" in chunk:
                service, file_path = chunk.split("::", 1)
                service = service.strip()
                file_path = file_path.strip()
                if service and file_path:
                    pairs.append((service, file_path))
                continue

            if len(services) == 1:
                pairs.append((services[0], chunk))

        return pairs

    def _canonicalize_source_pairs(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        canonical_pairs: set[tuple[str, str]] = set()

        for service, raw_file_path in pairs:
            clean_service = service.strip()
            clean_file_path = self._canonicalize_path(raw_file_path, clean_service)
            if clean_service and clean_file_path:
                canonical_pairs.add((clean_service, clean_file_path))

        return sorted(canonical_pairs)

    @staticmethod
    def _canonicalize_path(raw_file_path: str, service: str) -> str:
        path = raw_file_path.strip().replace("\\", "/")
        while "//" in path:
            path = path.replace("//", "/")
        if path.startswith("./"):
            path = path[2:]
        service_prefix = f"summaries/{service}/"
        if path.startswith(service_prefix):
            path = path[len(service_prefix):]
        return path

    @staticmethod
    def _build_doc_id(service: str, file_path: str) -> str:
        return hashlib.sha256(f"{service}::{file_path}".encode("utf-8")).hexdigest()

    @staticmethod
    def _dedupe_qrels(qrels: list[QrelRecord]) -> list[QrelRecord]:
        seen: set[tuple[str, str]] = set()
        deduped: list[QrelRecord] = []
        for qrel in qrels:
            key = (qrel.query_id, qrel.doc_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(qrel)
        return deduped

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
