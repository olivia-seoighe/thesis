"""HTTP strategy runner for retrieval endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class StrategyResponseChunk:
    """A normalized chunk returned by a strategy endpoint."""

    chunk_id: str
    document_id: str
    score: float
    source: str
    metadata: dict[str, Any]
    model_used: str
    keyword_ranker: str
    search_duration_ms: float


class StrategyRunner:
    """Calls retrieval endpoints and normalizes chunk responses."""

    ENDPOINTS = {
        "keyword": "/search/keyword",
        "keyword-fts": "/search/keyword",
        "keyword-bm25": "/search/keyword",
        "vector": "/search/vector",
        "hybrid": "/search/hybrid",
        "hybrid-fts": "/search/hybrid",
        "hybrid-bm25": "/search/hybrid",
        "hybrid-bm25-structured-first": "/search/hybrid",
        "graph": "/search/graph",
        "graph-adaptive": "/search/graph",
        "graph-fixed": "/search/graph",
    }

    def __init__(self, base_url: str, timeout_seconds: int = 60, retrieval_corpus: str = "summaries") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retrieval_corpus = retrieval_corpus

    def search(
        self,
        strategy: str,
        query_text: str,
        top_k: int,
        sources: tuple[str, ...] = (),
        retrieval_corpus: str = "summaries",
    ) -> list[StrategyResponseChunk]:
        endpoint = self.ENDPOINTS.get(strategy)
        if endpoint is None:
            raise ValueError(f"Unsupported strategy: {strategy}")

        params: dict[str, Any] = {
            "query": query_text,
            "top_k": top_k,
        }
        if sources:
            params["source"] = ",".join(sources)
        params["corpus"] = retrieval_corpus
        if strategy in {"graph", "graph-fixed"}:
            params["hop_policy"] = "fixed"
        elif strategy == "graph-adaptive":
            params["hop_policy"] = "adaptive"
        elif strategy in {"keyword-fts", "hybrid-fts"}:
            params["keyword_ranker"] = "fts"
        elif strategy in {"keyword-bm25", "hybrid-bm25", "hybrid-bm25-structured-first"}:
            params["keyword_ranker"] = "bm25"
        if strategy == "hybrid-bm25-structured-first":
            params["structured_first"] = "true"

        url = f"{self.base_url}{endpoint}?{urlencode(params)}"
        request = Request(url=url, method="GET")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} from {url}. Check retrieval-url and endpoint availability."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Failed to reach retrieval URL {self.base_url}: {exc.reason}."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timed out waiting for {url} after {self.timeout_seconds}s."
            ) from exc

        return self._extract_chunks(payload, strategy)

    def list_sources(self, retrieval_corpus: str | None = None) -> tuple[str, ...]:
        corpus = retrieval_corpus or self.retrieval_corpus
        url = f"{self.base_url}/sources?{urlencode({'corpus': corpus})}"
        request = Request(url=url, method="GET")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} from {url}. Cannot build service catalogue."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Failed to reach retrieval URL {self.base_url}: {exc.reason}."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timed out waiting for {url} after {self.timeout_seconds}s."
            ) from exc

        if not isinstance(payload, dict):
            return ()
        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            return ()
        sources = [str(source).strip() for source in raw_sources if str(source).strip()]
        return tuple(sorted(set(sources)))

    @staticmethod
    def _extract_chunks(payload: Any, strategy: str) -> list[StrategyResponseChunk]:
        chunks: list[StrategyResponseChunk] = []
        if not isinstance(payload, list):
            return chunks

        for response_row in payload:
            if not isinstance(response_row, dict):
                continue
            response_model_used = str(response_row.get("model_used", "")).strip()
            response_search_duration_ms = float(response_row.get("search_duration_ms", 0.0) or 0.0)
            row_chunks = response_row.get("chunks", [])
            if not isinstance(row_chunks, list):
                continue
            for chunk in row_chunks:
                parsed = StrategyRunner._extract_chunk(
                    chunk,
                    response_model_used=response_model_used,
                    response_search_duration_ms=response_search_duration_ms,
                    strategy=strategy,
                )
                if parsed is not None:
                    chunks.append(parsed)
        return chunks

    @staticmethod
    def _extract_chunk(
        chunk: Any,
        *,
        response_model_used: str,
        response_search_duration_ms: float,
        strategy: str,
    ) -> StrategyResponseChunk | None:
        if not isinstance(chunk, dict):
            return None
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        document_id = str(chunk.get("document_id", "")).strip()
        if not chunk_id or not document_id:
            return None
        score = float(chunk.get("score", 0.0))
        source = str(chunk.get("source", "")).strip()
        metadata_raw = chunk.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        ranker_value = metadata.get("keyword_ranker")
        keyword_ranker = str(ranker_value).strip() if isinstance(ranker_value, str) else ""
        if not keyword_ranker:
            if strategy in {"keyword-bm25", "hybrid-bm25", "hybrid-bm25-structured-first"}:
                keyword_ranker = "bm25"
            elif strategy in {"keyword", "keyword-fts", "hybrid", "hybrid-fts"}:
                keyword_ranker = "fts"

        return StrategyResponseChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            score=score,
            source=source,
            metadata=metadata,
            model_used=response_model_used,
            keyword_ranker=keyword_ranker,
            search_duration_ms=response_search_duration_ms,
        )
