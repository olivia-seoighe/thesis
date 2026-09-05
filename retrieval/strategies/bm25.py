"""Pure Okapi BM25 retrieval over chunk-level corpus."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from retrieval.utils.corpus import normalize_retrieval_corpus

# Tokenization preserves software identifiers like order-service.
_BM25_TOKEN = re.compile(r"\w+(?:-\w+)*")


@dataclass(frozen=True)
class BM25Document:
    chunk_id: str
    text: str
    source_code: str
    document_id: str
    document_title: str
    url: str
    metadata: dict[str, Any]
    source: str
    retrieval_corpus: str
    last_modified_date: str
    tokens: tuple[str, ...]
    term_freq: dict[str, int]
    doc_len: int


@dataclass(frozen=True)
class BM25SearchHit:
    document: BM25Document
    score: float


@dataclass(frozen=True)
class BM25BuildStats:
    document_count: int
    avgdl: float
    build_duration_ms: float


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 scoring."""
    return [token.lower() for token in _BM25_TOKEN.findall(text)]


class BM25Index:
    """In-memory BM25 index with lazy, cached corpus statistics."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._build_lock = asyncio.Lock()
        self._is_built = False

        self._documents: list[BM25Document] = []
        self._postings: dict[str, list[int]] = {}
        self._doc_freq: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._total_docs: int = 0

        self._source_to_indices: dict[str, list[int]] = {}
        self._source_doc_counts: dict[str, int] = {}
        self._source_total_lengths: dict[str, int] = {}

        self._last_build_duration_ms: float = 0.0
        self._last_built_at_epoch: float | None = None

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def last_build_duration_ms(self) -> float:
        return self._last_build_duration_ms

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def invalidate(self) -> None:
        self._is_built = False
        self._documents = []
        self._postings = {}
        self._doc_freq = {}
        self._avgdl = 0.0
        self._total_docs = 0
        self._source_to_indices = {}
        self._source_doc_counts = {}
        self._source_total_lengths = {}

    async def ensure_built(
        self,
        fetch_rows: Callable[[], Awaitable[list[Mapping[str, Any]]]],
    ) -> BM25BuildStats:
        if self._is_built:
            return BM25BuildStats(
                document_count=self._total_docs,
                avgdl=self._avgdl,
                build_duration_ms=0.0,
            )

        async with self._build_lock:
            if self._is_built:
                return BM25BuildStats(
                    document_count=self._total_docs,
                    avgdl=self._avgdl,
                    build_duration_ms=0.0,
                )

            start = time.time()
            rows = await fetch_rows()
            self._build_from_rows(rows)
            duration_ms = (time.time() - start) * 1000
            self._last_build_duration_ms = duration_ms
            self._last_built_at_epoch = time.time()
            return BM25BuildStats(
                document_count=self._total_docs,
                avgdl=self._avgdl,
                build_duration_ms=duration_ms,
            )

    async def rebuild(
        self,
        fetch_rows: Callable[[], Awaitable[list[Mapping[str, Any]]]],
    ) -> BM25BuildStats:
        self.invalidate()
        return await self.ensure_built(fetch_rows)

    def search(
        self,
        *,
        query: str,
        top_k: int,
        sources: Sequence[str] | None,
        retrieval_corpus: str,
        max_chunks_per_document: int | None,
        match_all: bool,
    ) -> list[BM25SearchHit]:
        if not self._is_built or top_k <= 0:
            return []

        query_terms = tokenize_bm25(query)
        if not query_terms:
            return []

        unique_terms = tuple(dict.fromkeys(query_terms))
        allowed_sources = {source.strip() for source in (sources or []) if source.strip()}
        try:
            corpus_token = normalize_retrieval_corpus(retrieval_corpus)
        except ValueError:
            corpus_token = "summaries"

        def _in_scope(document: BM25Document) -> bool:
            if allowed_sources and document.source not in allowed_sources:
                return False
            if corpus_token == "all":
                return True
            document_corpus = (document.retrieval_corpus or "summaries").strip().lower()
            return document_corpus == corpus_token

        scoped_documents = [document for document in self._documents if _in_scope(document)]
        corpus_size = len(scoped_documents)
        avgdl = (
            sum(document.doc_len for document in scoped_documents) / corpus_size
            if corpus_size
            else 0.0
        )

        if corpus_size == 0 or avgdl <= 0:
            return []

        term_doc_freq_in_scope: dict[str, int] = {}
        for term in unique_terms:
            if allowed_sources:
                postings = self._postings.get(term, [])
                term_doc_freq_in_scope[term] = sum(
                    1
                    for idx in postings
                    if _in_scope(self._documents[idx])
                )
            else:
                if corpus_token == "all":
                    term_doc_freq_in_scope[term] = self._doc_freq.get(term, 0)
                else:
                    term_doc_freq_in_scope[term] = sum(
                        1
                        for idx in self._postings.get(term, [])
                        if _in_scope(self._documents[idx])
                    )

        candidate_indices: set[int] = set()
        for term in unique_terms:
            for idx in self._postings.get(term, []):
                document = self._documents[idx]
                if not _in_scope(document):
                    continue
                candidate_indices.add(idx)

        if not candidate_indices:
            return []

        if match_all:
            doc_coverage: dict[str, set[str]] = {}
            for term in unique_terms:
                for idx in self._postings.get(term, []):
                    doc = self._documents[idx]
                    if not _in_scope(doc):
                        continue
                    doc_coverage.setdefault(doc.document_id, set()).add(term)
            qualifying_docs = {
                doc_id
                for doc_id, covered in doc_coverage.items()
                if len(covered) == len(unique_terms)
            }
            candidate_indices = {
                idx for idx in candidate_indices if self._documents[idx].document_id in qualifying_docs
            }
            if not candidate_indices:
                return []

        scored_hits: list[BM25SearchHit] = []
        for idx in candidate_indices:
            doc = self._documents[idx]
            score = 0.0
            for term in query_terms:
                freq = doc.term_freq.get(term, 0)
                if freq == 0:
                    continue
                n_q = term_doc_freq_in_scope.get(term, 0)
                if n_q <= 0:
                    continue
                idf = math.log(1.0 + (corpus_size - n_q + 0.5) / (n_q + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc.doc_len / avgdl))
                score += idf * ((freq * (self.k1 + 1.0)) / denom)

            if score > 0.0:
                scored_hits.append(BM25SearchHit(document=doc, score=score))

        scored_hits.sort(key=lambda item: (item.score, item.document.chunk_id), reverse=True)

        ranked_hits: list[BM25SearchHit] = []
        per_document_counts: dict[str, int] = {}
        for hit in scored_hits:
            doc_id = hit.document.document_id
            if max_chunks_per_document is not None:
                current = per_document_counts.get(doc_id, 0)
                if current >= max_chunks_per_document:
                    continue
                per_document_counts[doc_id] = current + 1
            ranked_hits.append(hit)
            if len(ranked_hits) >= top_k:
                break

        return ranked_hits

    def _build_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        documents: list[BM25Document] = []
        postings_sets: dict[str, set[int]] = {}
        source_to_indices: dict[str, list[int]] = {}
        source_doc_counts: dict[str, int] = {}
        source_total_lengths: dict[str, int] = {}

        total_len = 0

        for row in rows:
            text = str(row.get("text") or "")
            document_title = str(row.get("document_title") or row.get("name") or "")
            combined_text = f"{text}\n{document_title}"
            tokens = tokenize_bm25(combined_text)
            term_counts = Counter(tokens)
            doc_len = len(tokens)
            source = str(row.get("source") or "")

            metadata_raw = row.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            document = BM25Document(
                chunk_id=str(row.get("chunk_id") or ""),
                text=text,
                source_code=str(row.get("source_code") or ""),
                document_id=str(row.get("document_id") or ""),
                document_title=document_title,
                url=str(row.get("url") or ""),
                metadata=metadata,
                source=source,
                retrieval_corpus=str(row.get("retrieval_corpus") or "summaries"),
                last_modified_date=str(row.get("last_modified_date") or ""),
                tokens=tuple(tokens),
                term_freq=dict(term_counts),
                doc_len=doc_len,
            )
            doc_idx = len(documents)
            documents.append(document)

            source_to_indices.setdefault(source, []).append(doc_idx)
            source_doc_counts[source] = source_doc_counts.get(source, 0) + 1
            source_total_lengths[source] = source_total_lengths.get(source, 0) + doc_len

            total_len += doc_len

            for term in term_counts:
                postings_sets.setdefault(term, set()).add(doc_idx)

        postings = {term: sorted(list(indices)) for term, indices in postings_sets.items()}
        doc_freq = {term: len(indices) for term, indices in postings.items()}

        self._documents = documents
        self._postings = postings
        self._doc_freq = doc_freq
        self._source_to_indices = source_to_indices
        self._source_doc_counts = source_doc_counts
        self._source_total_lengths = source_total_lengths
        self._total_docs = len(documents)
        self._avgdl = (total_len / self._total_docs) if self._total_docs else 0.0
        self._is_built = True
