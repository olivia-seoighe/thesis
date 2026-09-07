"""Source-aware search client backed by PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Optional

import asyncpg

from retrieval.clients.embedding_client import EmbeddingAPIClient
from retrieval.models.models import RetrievedChunk, SearchRequest, SearchResponse
from retrieval.strategies.bm25 import BM25Index, tokenize_bm25
from retrieval.utils.logging_config import get_logger

logger = get_logger(__name__)


def _term_to_tsquery(term: str) -> str:
    """Convert one search term into PostgreSQL tsquery phrase syntax."""
    safe_term = term.replace("'", "''")
    words = safe_term.split()
    if len(words) > 1:
        return " <-> ".join(words)
    return safe_term


# Preserve hyphenated identifiers (e.g., service IDs like order-service).
_TSQUERY_TOKEN = re.compile(r"\w+(?:-\w+)*")


# Extracts PostgreSQL-safe keyword terms while preserving service-style names.
def _sanitize_terms(query: str) -> list[str]:
    return _TSQUERY_TOKEN.findall(query)


class SearchClient:
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        min_size: int = 2,
        max_size: int = 5,
        **pool_kwargs: Any,
    ):
        self.host = host or os.getenv("PGHOST")
        self.port = port or int(os.getenv("PGPORT", "5432"))
        self.database = database or os.getenv("PGDATABASE")
        self.user = user or os.getenv("PGUSER")
        self.password = password or os.getenv("PGPASSWORD")

        self.min_size = int(os.getenv("PGPOOL_MIN_SIZE", str(min_size)))
        self.max_size = int(os.getenv("PGPOOL_MAX_SIZE", str(max_size)))

        if self.host and "database.azure.com" in self.host:
            pool_kwargs.setdefault("ssl", "require")

        self.pool_kwargs = pool_kwargs
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_loop: Optional[asyncio.AbstractEventLoop] = None

        self.embedding_client = EmbeddingAPIClient()
        self.hnsw_ef_search = int(os.getenv("HNSW_EF_SEARCH", "400"))
        self.bm25_index = BM25Index(
            k1=float(os.getenv("BM25_K1", "1.2")),
            b=float(os.getenv("BM25_B", "0.75")),
        )

        logger.info(
            f"Initialized SearchClient: {self.host}:{self.port}/{self.database}, "
            f"hnsw_ef_search={self.hnsw_ef_search}"
        )

    async def _connection_init(self, conn: asyncpg.Connection) -> None:
        """Initialize a new pooled connection with JSON codecs."""
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )
        await conn.set_type_codec(
            "json",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )

    async def _connection_setup(self, conn: asyncpg.Connection) -> None:
        """Apply AGE and vector search settings before each connection use."""
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await conn.execute(f"SET hnsw.ef_search = {self.hnsw_ef_search};")

    async def get_pool(self) -> asyncpg.Pool:
        current_loop = asyncio.get_running_loop()
        if self._pool is None or self._pool_loop is not current_loop:
            if self._pool:
                await self._pool.close()

            logger.info(
                f"Creating connection pool: {self.host}:{self.port}/{self.database} "
                f"(min_size={self.min_size}, max_size={self.max_size})"
            )

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
                init=self._connection_init,
                setup=self._connection_setup,
                **self.pool_kwargs,
            )
            self._pool_loop = current_loop
            logger.info("Connection pool created successfully")

        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def executemany(self, query: str, args: list[tuple]) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(query, args)

    # Formats the source list consistently for API responses.
    @staticmethod
    def _source_searched(request: SearchRequest) -> str:
        return ",".join(request.sources) if request.sources else "all"

    # Normalizes asyncpg JSON/JSONB metadata into a mutable dict.
    @staticmethod
    def _metadata(row: asyncpg.Record | dict, *, source: str) -> dict:
        raw_metadata = row["metadata"]
        metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata or {}
        return {**metadata, "source": source}

    # Converts a database row into the API chunk model.
    @classmethod
    def _chunk_from_row(
        cls,
        row: asyncpg.Record | dict,
        *,
        metadata: dict,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            source_code=row["source_code"] or "",
            document_id=row["document_id"],
            document_title=row["name"] or "",
            url=row["url"] or "",
            last_modified_date=str(row["last_modified_date"]),
            metadata=metadata,
            source=row["source"],
            score=float(row["score"]),
        )

    # Builds vector SQL for source, corpus, and optional entity filters.
    @staticmethod
    def _vector_query(*, source: str | None, retrieval_corpus: str, entity_filter: str | None) -> tuple[str, list[Any]]:
        params: list[Any] = []

        def add_param(value: Any) -> str:
            params.append(value)
            return f"${len(params)}"

        vector_param = add_param("__VECTOR__")
        where = ["embedding_3072 IS NOT NULL"]
        if source is not None:
            where.append(f"source = {add_param(source)}")
        if retrieval_corpus != "all":
            where.append(f"retrieval_corpus = {add_param(retrieval_corpus)}")
        limit_param = add_param("__TOP_K__")

        if entity_filter:
            pattern_param = add_param(f"%{entity_filter}%")
            tsquery_param = add_param(entity_filter.lower().replace(" ", " & "))
            where.append(f"(document_title ILIKE {pattern_param} OR tsv @@ to_tsquery('english', {tsquery_param}))")
            score_select = f"1 - (embedding_3072 <=> {vector_param}::halfvec) AS score"
            cte_name = "filtered"
            cte_order = "score DESC"
            final_score = "f.score"
            final_order = "f.score DESC"
            final_alias = "f"
        else:
            score_select = f"embedding_3072 <=> {vector_param}::halfvec AS distance"
            cte_name = "nearest"
            cte_order = f"embedding_3072 <=> {vector_param}::halfvec ASC"
            final_score = "1 - n.distance AS score"
            final_order = "n.distance ASC"
            final_alias = "n"

        sql = f"""
            WITH {cte_name} AS (
                SELECT chunk_id, {score_select}
                FROM document_embeddings
                WHERE {' AND '.join(where)}
                ORDER BY {cte_order}
                LIMIT {limit_param}
            )
            SELECT
                de.chunk_id, de.text, de.source_code, de.document_id,
                dm.name, dm.url, (de.metadata)::jsonb AS metadata,
                de.source, dm.last_modified_date, {final_score}
            FROM {cte_name} {final_alias}
            JOIN document_embeddings de ON de.chunk_id = {final_alias}.chunk_id
            LEFT JOIN document_metadata dm ON de.document_id = dm.document_id
                AND de.retrieval_corpus = dm.retrieval_corpus
            ORDER BY {final_order};
        """
        return sql, params

    # Runs one vector query branch for either a specific source or all sources.
    async def _vector_search_once(
        self,
        conn: asyncpg.Connection,
        *,
        query_vector_text: str,
        source: str | None,
        retrieval_corpus: str,
        entity_filter: str | None,
        top_k: int,
    ) -> list[asyncpg.Record]:
        sql, params = self._vector_query(
            source=source,
            retrieval_corpus=retrieval_corpus,
            entity_filter=entity_filter,
        )
        params = [query_vector_text if value == "__VECTOR__" else top_k if value == "__TOP_K__" else value for value in params]
        return list(await conn.fetch(sql, *params))

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._pool_loop = None
            logger.info("Connection pool closed")

    async def test_connection(self) -> bool:
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform vector similarity search using HNSW index."""
        start_time = time.time()
        model_name = self.embedding_client.default_model

        embedding_start = time.time()
        query_embedding = await self.embedding_client.embed_single(request.query)
        embedding_duration = (time.time() - embedding_start) * 1000
        query_vector_text = "[" + ",".join(str(v) for v in query_embedding) + "]"

        try:
            pool = await self.get_pool()
            all_rows: list[asyncpg.Record] = []

            async with pool.acquire() as conn:
                search_start = time.time()
                for source in request.sources or [None]:
                    rows = await self._vector_search_once(
                        conn,
                        query_vector_text=query_vector_text,
                        source=source,
                        retrieval_corpus=request.retrieval_corpus,
                        entity_filter=request.entity_filter,
                        top_k=request.top_k,
                    )
                    all_rows.extend(rows)
                    logger.debug("Vector branch returned %s results", len(rows))
                search_duration = (time.time() - search_start) * 1000

            all_rows.sort(key=lambda r: r["score"], reverse=True)
            top_rows = all_rows[: request.top_k]
            chunks = [
                self._chunk_from_row(
                    row,
                    metadata={**self._metadata(row, source=row["source"]), "embedding_model": model_name},
                )
                for row in top_rows
            ]

            total_duration = (time.time() - start_time) * 1000
            logger.info(
                "Vector search completed",
                extra={
                    "search": {
                        "results_count": len(chunks),
                        "total_duration_ms": total_duration,
                        "embedding_duration_ms": embedding_duration,
                        "search_duration_ms": search_duration,
                        "sources": request.sources or "all",
                        "retrieval_corpus": request.retrieval_corpus,
                        "entity_filter": request.entity_filter,
                    }
                },
            )

            return SearchResponse(
                chunks=chunks,
                total_results=len(chunks),
                search_duration_ms=total_duration,
                embedding_duration_ms=embedding_duration,
                model_used=model_name,
                source_searched=self._source_searched(request),
            )

        except Exception as e:
            logger.error(
                "Error during vector search",
                extra={"error": str(e), "sources": request.sources},
                exc_info=True,
            )
            raise

    # Builds the document INTERSECT clause for cross-chunk all-term matching.
    @staticmethod
    def _match_all_intersect_sql(query_concepts: list[str], *, source: str | None) -> str:
        source_predicate = "source = $2" if source is not None else ""
        corpus_predicate = "retrieval_corpus = $3" if source is not None else "retrieval_corpus = $2"
        parts: list[str] = []
        for concept in query_concepts:
            where_predicates = [corpus_predicate, f"tsv @@ to_tsquery('english', '{_term_to_tsquery(concept)}')"]
            if source_predicate:
                where_predicates.insert(0, source_predicate)
            parts.append("SELECT document_id FROM document_embeddings " f"WHERE {' AND '.join(where_predicates)}")
        return " INTERSECT ".join(parts)

    # Builds the optional per-document chunk cap for match-all keyword search.
    @staticmethod
    def _match_all_tail_sql(*, source: str | None, max_chunks_per_document: int | None) -> str:
        top_k_param = "$4" if source is not None else "$3"
        max_chunks_param = "$5" if source is not None else "$4"
        if max_chunks_per_document is None:
            return f"""
                SELECT chunk_id, text, source_code, document_id, name, url, metadata,
                       source, last_modified_date, score, term_count, base_score
                FROM scored_chunks
                ORDER BY score DESC
                LIMIT {top_k_param};"""
        return f""",
                ranked_chunks AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY document_id ORDER BY score DESC
                    ) AS doc_rank
                    FROM scored_chunks
                )
                SELECT chunk_id, text, source_code, document_id, name, url, metadata,
                       source, last_modified_date, score, term_count, base_score
                FROM ranked_chunks
                WHERE doc_rank <= {max_chunks_param}
                ORDER BY score DESC
                LIMIT {top_k_param};"""

    # Orders match-all query parameters to match the generated SQL placeholders.
    @staticmethod
    def _match_all_params(
        *,
        tsquery_or: str,
        source: str | None,
        retrieval_corpus: str,
        top_k: int,
        max_chunks_per_document: int | None,
    ) -> tuple[Any, ...]:
        params: list[Any] = [tsquery_or]
        if source is not None:
            params.append(source)
        params.extend([retrieval_corpus, top_k])
        if max_chunks_per_document is not None:
            params.append(max_chunks_per_document)
        return tuple(params)

    async def _keyword_search_match_all(
        self,
        *,
        conn: asyncpg.Connection,
        source: str | None,
        retrieval_corpus: str,
        query_concepts: list[str],
        top_k: int,
        max_chunks_per_document: int | None,
    ) -> list[asyncpg.Record]:
        """Find documents where all concepts appear across their chunks."""
        qualifying_docs_sql = self._match_all_intersect_sql(query_concepts, source=source)
        tsquery_parts = [_term_to_tsquery(concept) for concept in query_concepts]
        tsquery_wrapped = [f"({p})" if " <-> " in p else p for p in tsquery_parts]
        tsquery_or = " | ".join(tsquery_wrapped)
        term_checks = [
            f"(de.tsv @@ to_tsquery('english', '{_term_to_tsquery(concept)}'))::int AS has_term_{index}"
            for index, concept in enumerate(query_concepts)
        ]
        term_check_sql = ", ".join(term_checks)
        total_terms = len(query_concepts)
        term_count_expr = " + ".join(f"has_term_{i}" for i in range(total_terms))

        logger.info(
            "Executing match_all keyword search (INTERSECT approach)",
            extra={"source": source or "all", "concepts": query_concepts, "top_k": top_k},
        )
        chunk_terms_source_filter = "AND de.source = $2" if source is not None else ""
        chunk_terms_corpus_filter = "AND de.retrieval_corpus = $3" if source is not None else "AND de.retrieval_corpus = $2"
        tail_sql = self._match_all_tail_sql(source=source, max_chunks_per_document=max_chunks_per_document)

        sql = f"""
            WITH qualifying_docs AS (
                {qualifying_docs_sql}
            ),
            chunk_terms AS (
                SELECT
                    de.chunk_id, de.text, de.source_code, de.document_id,
                    dm.name, dm.url,
                    (de.metadata)::jsonb AS metadata,
                    de.source, dm.last_modified_date,
                    ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS base_score,
                    {term_check_sql}
                FROM document_embeddings de
                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                JOIN qualifying_docs qd ON de.document_id = qd.document_id
                WHERE de.tsv @@ to_tsquery('english', $1)
                  {chunk_terms_source_filter}
                                    {chunk_terms_corpus_filter}
            ),
            scored_chunks AS (
                SELECT
                    chunk_id, text, source_code, document_id, name, url, metadata,
                    source, last_modified_date, base_score,
                    ({term_count_expr}) AS term_count,
                    base_score * (1.0 + 0.5 * ({term_count_expr})::float / {total_terms}) AS score
                FROM chunk_terms
            ){tail_sql}"""

        params = self._match_all_params(
            tsquery_or=tsquery_or,
            source=source,
            retrieval_corpus=retrieval_corpus,
            top_k=top_k,
            max_chunks_per_document=max_chunks_per_document,
        )
        return list(await conn.fetch(sql, *params))

    # Runs non-match-all PostgreSQL FTS for either a source or all sources.
    async def _keyword_search_any(
        self,
        *,
        conn: asyncpg.Connection,
        source: str | None,
        retrieval_corpus: str,
        tsquery_str: str,
        top_k: int,
        max_chunks_per_document: int | None,
    ) -> list[asyncpg.Record]:
        params: list[Any] = [tsquery_str]

        def add_param(value: Any) -> str:
            params.append(value)
            return f"${len(params)}"

        where = ["de.tsv IS NOT NULL", "de.tsv @@ to_tsquery('english', $1)"]
        if source is not None:
            where.append(f"de.source = {add_param(source)}")
        where.append(f"de.retrieval_corpus = {add_param(retrieval_corpus)}")
        top_k_param = add_param(top_k)

        if max_chunks_per_document is not None:
            max_chunks_param = add_param(max_chunks_per_document)
            sql = f"""
                WITH scored_chunks AS (
                    SELECT
                        de.chunk_id, de.text, de.source_code, de.document_id,
                        dm.name, dm.url, (de.metadata)::jsonb AS metadata,
                        de.source, dm.last_modified_date,
                        ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                    FROM document_embeddings de
                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id
                        AND de.retrieval_corpus = dm.retrieval_corpus
                    WHERE {' AND '.join(where)}
                ),
                ranked_chunks AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY document_id ORDER BY score DESC
                    ) AS doc_rank
                    FROM scored_chunks
                )
                SELECT chunk_id, text, source_code, document_id, name, url,
                       metadata, source, last_modified_date, score
                FROM ranked_chunks
                WHERE doc_rank <= {max_chunks_param}
                ORDER BY score DESC
                LIMIT {top_k_param};
            """
        else:
            sql = f"""
                SELECT
                    de.chunk_id, de.text, de.source_code, de.document_id,
                    dm.name, dm.url, (de.metadata)::jsonb AS metadata,
                    de.source, dm.last_modified_date,
                    ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                FROM document_embeddings de
                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id
                    AND de.retrieval_corpus = dm.retrieval_corpus
                WHERE {' AND '.join(where)}
                ORDER BY score DESC
                LIMIT {top_k_param};
            """
        return list(await conn.fetch(sql, *params))

    async def _fetch_bm25_corpus_rows(self) -> list[asyncpg.Record]:
        """Fetch chunk corpus used by BM25 at the same retrieval unit (chunk)."""
        sql = """
            SELECT
                de.chunk_id,
                de.text,
                de.source_code,
                de.document_id,
                COALESCE(de.document_title, dm.name, '') AS document_title,
                dm.name,
                dm.url,
                (de.metadata)::jsonb AS metadata,
                de.source,
                de.retrieval_corpus,
                dm.last_modified_date
            FROM document_embeddings de
            LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus;
        """
        return await self.fetch(sql)

    def invalidate_bm25_index(self) -> None:
        self.bm25_index.invalidate()

    async def rebuild_bm25_index(self) -> None:
        stats = await self.bm25_index.rebuild(self._fetch_bm25_corpus_rows)
        logger.info(
            "BM25 index rebuilt",
            extra={
                "bm25": {
                    "documents": stats.document_count,
                    "avgdl": stats.avgdl,
                    "build_duration_ms": stats.build_duration_ms,
                }
            },
        )

    async def search_keyword_fts(self, request: SearchRequest) -> SearchResponse:
        """Perform keyword search using PostgreSQL FTS + ts_rank_cd."""
        start_time = time.time()
        terms = _sanitize_terms(request.query)

        try:
            pool = await self.get_pool()
            all_rows: list[dict] = []
            tsquery_str = " | ".join(terms)

            async with pool.acquire() as conn:
                search_start = time.time()
                for source in request.sources or [None]:
                    if request.match_all:
                        rows = await self._keyword_search_match_all(
                            conn=conn,
                            source=source,
                            retrieval_corpus=request.retrieval_corpus,
                            query_concepts=terms,
                            top_k=request.top_k,
                            max_chunks_per_document=request.max_chunks_per_document,
                        )
                    else:
                        rows = await self._keyword_search_any(
                            conn=conn,
                            source=source,
                            retrieval_corpus=request.retrieval_corpus,
                            tsquery_str=tsquery_str,
                            top_k=request.top_k,
                            max_chunks_per_document=request.max_chunks_per_document,
                        )
                    all_rows.extend([dict(row) for row in rows])
                    logger.debug("Keyword branch returned %s results", len(rows))
                search_duration = (time.time() - search_start) * 1000

            all_rows.sort(key=lambda r: r["score"], reverse=True)
            chunks: list[RetrievedChunk] = []
            for row in all_rows[: request.top_k]:
                chunk_metadata = {
                    **self._metadata(row, source=row["source"]),
                    "search_type": "keyword",
                    "keyword_ranker": "fts_ts_rank_cd",
                }
                if request.match_all and "term_count" in row:
                    chunk_metadata["match_all"] = {
                        "document_qualifies": True,
                        "match_mode": "cross_chunk",
                        "term_coverage": {
                            "terms_in_chunk": row["term_count"],
                            "base_score": float(row.get("base_score", 0)),
                            "boosted_score": float(row["score"]),
                        },
                    }
                chunks.append(self._chunk_from_row(row, metadata=chunk_metadata))

            total_duration = (time.time() - start_time) * 1000
            logger.info(
                "Keyword search completed",
                extra={
                    "search": {
                        "results_count": len(chunks),
                        "unique_documents": len({c.document_id for c in chunks}),
                        "total_duration_ms": total_duration,
                        "search_duration_ms": search_duration,
                        "sources": request.sources or "all",
                        "retrieval_corpus": request.retrieval_corpus,
                        "match_all": request.match_all,
                    }
                },
            )
            return SearchResponse(
                chunks=chunks,
                total_results=len(chunks),
                search_duration_ms=total_duration,
                embedding_duration_ms=0.0,
                model_used="postgresql-fts",
                source_searched=self._source_searched(request),
            )

        except Exception as e:
            logger.error(
                "Error during keyword search",
                extra={"error": str(e), "sources": request.sources},
                exc_info=True,
            )
            raise

    async def search_keyword_bm25(self, request: SearchRequest) -> SearchResponse:
        """Perform pure Okapi BM25 keyword search over cached chunk corpus."""
        terms = tokenize_bm25(request.query)
        if not terms:
            return SearchResponse(
                chunks=[],
                total_results=0,
                search_duration_ms=0.0,
                embedding_duration_ms=0.0,
                model_used="okapi-bm25",
                source_searched=self._source_searched(request),
            )

        build_stats = await self.bm25_index.ensure_built(self._fetch_bm25_corpus_rows)
        if build_stats.build_duration_ms > 0:
            logger.info(
                "BM25 index built",
                extra={
                    "bm25": {
                        "documents": build_stats.document_count,
                        "avgdl": build_stats.avgdl,
                        "build_duration_ms": build_stats.build_duration_ms,
                    }
                },
            )

        search_start = time.time()
        hits = self.bm25_index.search(
            query=request.query,
            top_k=request.top_k,
            sources=request.sources,
            retrieval_corpus=request.retrieval_corpus,
            max_chunks_per_document=request.max_chunks_per_document,
            match_all=request.match_all,
        )
        search_duration = (time.time() - search_start) * 1000

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            doc = hit.document
            chunk_metadata: dict = {
                **(doc.metadata or {}),
                "search_type": "keyword",
                "keyword_ranker": "bm25",
                "bm25_k1": self.bm25_index.k1,
                "bm25_b": self.bm25_index.b,
                "source": doc.source,
            }
            if request.match_all:
                chunk_metadata["match_all"] = {
                    "document_qualifies": True,
                    "match_mode": "cross_chunk_filter_only",
                }

            chunks.append(
                RetrievedChunk(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    source_code=doc.source_code,
                    document_id=doc.document_id,
                    document_title=doc.document_title,
                    url=doc.url,
                    last_modified_date=doc.last_modified_date,
                    metadata=chunk_metadata,
                    source=doc.source,
                    score=float(hit.score),
                )
            )

        logger.info(
            "BM25 keyword search completed",
            extra={
                "search": {
                    "results_count": len(chunks),
                    "search_duration_ms": search_duration,
                    "sources": request.sources or "all",
                    "retrieval_corpus": request.retrieval_corpus,
                    "match_all": request.match_all,
                    "k1": self.bm25_index.k1,
                    "b": self.bm25_index.b,
                }
            },
        )

        return SearchResponse(
            chunks=chunks,
            total_results=len(chunks),
            search_duration_ms=search_duration,
            embedding_duration_ms=0.0,
            model_used="okapi-bm25",
            source_searched=self._source_searched(request),
        )

    async def search_keyword(self, request: SearchRequest) -> SearchResponse:
        """Dispatch keyword search by configured ranker."""
        if request.keyword_ranker == "bm25":
            return await self.search_keyword_bm25(request)
        return await self.search_keyword_fts(request)

    async def health_check(self) -> bool:
        try:
            await self.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False