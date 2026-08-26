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


class StrategyRunner:
    """Calls retrieval endpoints and normalizes chunk responses."""

    ENDPOINTS = {
        "keyword": "/search/keyword",
        "vector": "/search/vector",
        "hybrid": "/search/hybrid",
        "graph": "/search/graph",
    }

    def __init__(self, base_url: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        strategy: str,
        query_text: str,
        top_k: int,
        sources: tuple[str, ...] = (),
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

        return self._extract_chunks(payload)

    def list_sources(self) -> tuple[str, ...]:
        url = f"{self.base_url}/sources"
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

        if not isinstance(payload, dict):
            return ()
        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            return ()
        sources = [str(source).strip() for source in raw_sources if str(source).strip()]
        return tuple(sorted(set(sources)))

    @staticmethod
    def _extract_chunks(payload: Any) -> list[StrategyResponseChunk]:
        chunks: list[StrategyResponseChunk] = []
        if not isinstance(payload, list):
            return chunks

        for response_row in payload:
            if not isinstance(response_row, dict):
                continue
            row_chunks = response_row.get("chunks", [])
            if not isinstance(row_chunks, list):
                continue
            for chunk in row_chunks:
                parsed = StrategyRunner._extract_chunk(chunk)
                if parsed is not None:
                    chunks.append(parsed)
        return chunks

    @staticmethod
    def _extract_chunk(chunk: Any) -> StrategyResponseChunk | None:
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
        return StrategyResponseChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            score=score,
            source=source,
            metadata=metadata,
        )
