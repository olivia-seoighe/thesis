"""Source-aware search client backed by PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, List, Optional

import asyncpg

from retrieval.clients.embedding_client import EmbeddingAPIClient
from retrieval.models.models import RetrievedChunk, SearchRequest, SearchResponse
from retrieval.strategies.bm25 import BM25Index, tokenize_bm25
from retrieval.utils.logging_config import get_logger

logger = get_logger(__name__)


def _term_to_tsquery(term: str) -> str:
    """Convert a term to tsquery format for phrase matching.

    Multi-word terms become phrase-matched ("service level agreement"
    → "service <-> level <-> agreement") using PostgreSQL's <-> operator.
    """
    safe_term = term.replace("'", "''")
    words = safe_term.split()
    if len(words) > 1:
        return " <-> ".join(words)
    return safe_term


# Preserve hyphenated identifiers (e.g., service IDs like order-service).
_TSQUERY_TOKEN = re.compile(r"\w+(?:-\w+)*")
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
        """Initialize a new connection with type codecs.

        Called once when a connection is first created in the pool.
        Type codecs persist across pool reuse since they are set on the
        asyncpg connection object, not via SQL SET commands.
        """
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
        """Configure session settings before each use.

        Called every time a connection is acquired from the pool. asyncpg
        runs RESET ALL when connections are returned, which clears prior SET
        commands. Using the pool's setup callback guarantees hnsw.ef_search
        is always applied before a query runs.
        """
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

                sources_to_search = request.sources or []
                retrieval_corpus = request.retrieval_corpus
                filter_by_corpus = retrieval_corpus != "all"

                if sources_to_search:
                    for source in sources_to_search:
                        query_params: list = [query_vector_text, source, request.top_k]

                        if request.entity_filter:
                            entity_pattern = f"%{request.entity_filter}%"
                            entity_tsquery = request.entity_filter.lower().replace(" ", " & ")
                            if filter_by_corpus:
                                query_params.extend([retrieval_corpus, entity_pattern, entity_tsquery])

                                sql = """
                                    WITH filtered AS (
                                        SELECT chunk_id,
                                               1 - (embedding_3072 <=> $1::halfvec) AS score
                                        FROM document_embeddings
                                        WHERE source = $2
                                          AND embedding_3072 IS NOT NULL
                                          AND retrieval_corpus = $4
                                          AND (
                                            document_title ILIKE $5
                                            OR tsv @@ to_tsquery('english', $6)
                                          )
                                        ORDER BY score DESC
                                        LIMIT $3
                                    )
                                    SELECT
                                        de.chunk_id,
                                        de.text,
                                        de.source_code,
                                        de.document_id,
                                        dm.name,
                                        dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source,
                                        dm.last_modified_date,
                                        f.score
                                    FROM filtered f
                                    JOIN document_embeddings de ON de.chunk_id = f.chunk_id
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    ORDER BY f.score DESC;
                                """
                            else:
                                query_params.extend([entity_pattern, entity_tsquery])

                                # Pre-filter by entity, compute distances on the filtered set.
                                # Uses computed score (not native distance ordering) to force
                                # exact scan -- HNSW iterative scan fails with selective entity
                                # filters that match few rows.
                                sql = """
                                    WITH filtered AS (
                                        SELECT chunk_id,
                                               1 - (embedding_3072 <=> $1::halfvec) AS score
                                        FROM document_embeddings
                                        WHERE source = $2
                                          AND embedding_3072 IS NOT NULL
                                          AND (
                                            document_title ILIKE $4
                                            OR tsv @@ to_tsquery('english', $5)
                                          )
                                        ORDER BY score DESC
                                        LIMIT $3
                                    )
                                    SELECT
                                        de.chunk_id,
                                        de.text,
                                        de.source_code,
                                        de.document_id,
                                        dm.name,
                                        dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source,
                                        dm.last_modified_date,
                                        f.score
                                    FROM filtered f
                                    JOIN document_embeddings de ON de.chunk_id = f.chunk_id
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    ORDER BY f.score DESC;
                                """
                        else:
                            if filter_by_corpus:
                                query_params.append(retrieval_corpus)
                                sql = """
                                    WITH nearest AS (
                                        SELECT chunk_id,
                                               embedding_3072 <=> $1::halfvec AS distance
                                        FROM document_embeddings
                                        WHERE source = $2
                                          AND embedding_3072 IS NOT NULL
                                          AND retrieval_corpus = $4
                                        ORDER BY embedding_3072 <=> $1::halfvec ASC
                                        LIMIT $3
                                    )
                                    SELECT
                                        de.chunk_id,
                                        de.text,
                                        de.source_code,
                                        de.document_id,
                                        dm.name,
                                        dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source,
                                        dm.last_modified_date,
                                        1 - n.distance AS score
                                    FROM nearest n
                                    JOIN document_embeddings de ON de.chunk_id = n.chunk_id
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    ORDER BY n.distance ASC;
                                """
                            else:
                                # HNSW-optimized: native distance ordering enables index scan.
                                # CTE avoids reading large columns for all candidates.
                                sql = """
                                    WITH nearest AS (
                                        SELECT chunk_id,
                                               embedding_3072 <=> $1::halfvec AS distance
                                        FROM document_embeddings
                                        WHERE source = $2
                                          AND embedding_3072 IS NOT NULL
                                        ORDER BY embedding_3072 <=> $1::halfvec ASC
                                        LIMIT $3
                                    )
                                    SELECT
                                        de.chunk_id,
                                        de.text,
                                        de.source_code,
                                        de.document_id,
                                        dm.name,
                                        dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source,
                                        dm.last_modified_date,
                                        1 - n.distance AS score
                                    FROM nearest n
                                    JOIN document_embeddings de ON de.chunk_id = n.chunk_id
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    ORDER BY n.distance ASC;
                                """

                        rows = await conn.fetch(sql, *query_params)
                        all_rows.extend(rows)
                        logger.debug(f"Source '{source}' returned {len(rows)} results")
                else:
                    # Full corpus search — no source filter
                    if request.entity_filter:
                        entity_pattern = f"%{request.entity_filter}%"
                        entity_tsquery = request.entity_filter.lower().replace(" ", " & ")
                        if filter_by_corpus:
                            query_params = [
                                query_vector_text,
                                request.top_k,
                                retrieval_corpus,
                                entity_pattern,
                                entity_tsquery,
                            ]
                            sql = """
                                WITH filtered AS (
                                    SELECT chunk_id,
                                           1 - (embedding_3072 <=> $1::halfvec) AS score
                                    FROM document_embeddings
                                    WHERE embedding_3072 IS NOT NULL
                                      AND retrieval_corpus = $3
                                      AND (
                                        document_title ILIKE $4
                                        OR tsv @@ to_tsquery('english', $5)
                                      )
                                    ORDER BY score DESC
                                    LIMIT $2
                                )
                                SELECT
                                    de.chunk_id,
                                    de.text,
                                    de.source_code,
                                    de.document_id,
                                    dm.name,
                                    dm.url,
                                    (de.metadata)::jsonb AS metadata,
                                    de.source,
                                    dm.last_modified_date,
                                    f.score
                                FROM filtered f
                                JOIN document_embeddings de ON de.chunk_id = f.chunk_id
                                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                ORDER BY f.score DESC;
                            """
                        else:
                            query_params = [query_vector_text, request.top_k, entity_pattern, entity_tsquery]
                            sql = """
                                WITH filtered AS (
                                    SELECT chunk_id,
                                           1 - (embedding_3072 <=> $1::halfvec) AS score
                                    FROM document_embeddings
                                    WHERE embedding_3072 IS NOT NULL
                                      AND (
                                        document_title ILIKE $3
                                        OR tsv @@ to_tsquery('english', $4)
                                      )
                                    ORDER BY score DESC
                                    LIMIT $2
                                )
                                SELECT
                                    de.chunk_id,
                                    de.text,
                                    de.source_code,
                                    de.document_id,
                                    dm.name,
                                    dm.url,
                                    (de.metadata)::jsonb AS metadata,
                                    de.source,
                                    dm.last_modified_date,
                                    f.score
                                FROM filtered f
                                JOIN document_embeddings de ON de.chunk_id = f.chunk_id
                                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                ORDER BY f.score DESC;
                            """
                    else:
                        if filter_by_corpus:
                            query_params = [query_vector_text, request.top_k, retrieval_corpus]
                            sql = """
                                WITH nearest AS (
                                    SELECT chunk_id,
                                           embedding_3072 <=> $1::halfvec AS distance
                                    FROM document_embeddings
                                    WHERE embedding_3072 IS NOT NULL
                                      AND retrieval_corpus = $3
                                    ORDER BY embedding_3072 <=> $1::halfvec ASC
                                    LIMIT $2
                                )
                                SELECT
                                    de.chunk_id,
                                    de.text,
                                    de.source_code,
                                    de.document_id,
                                    dm.name,
                                    dm.url,
                                    (de.metadata)::jsonb AS metadata,
                                    de.source,
                                    dm.last_modified_date,
                                    1 - n.distance AS score
                                FROM nearest n
                                JOIN document_embeddings de ON de.chunk_id = n.chunk_id
                                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                ORDER BY n.distance ASC;
                            """
                        else:
                            query_params = [query_vector_text, request.top_k]
                            sql = """
                                WITH nearest AS (
                                    SELECT chunk_id,
                                           embedding_3072 <=> $1::halfvec AS distance
                                    FROM document_embeddings
                                    WHERE embedding_3072 IS NOT NULL
                                    ORDER BY embedding_3072 <=> $1::halfvec ASC
                                    LIMIT $2
                                )
                                SELECT
                                    de.chunk_id,
                                    de.text,
                                    de.source_code,
                                    de.document_id,
                                    dm.name,
                                    dm.url,
                                    (de.metadata)::jsonb AS metadata,
                                    de.source,
                                    dm.last_modified_date,
                                    1 - n.distance AS score
                                FROM nearest n
                                JOIN document_embeddings de ON de.chunk_id = n.chunk_id
                                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                ORDER BY n.distance ASC;
                            """
                    rows = await conn.fetch(sql, *query_params)
                    all_rows.extend(rows)
                    logger.debug(f"Full corpus search returned {len(rows)} results")

                search_duration = (time.time() - search_start) * 1000

            all_rows.sort(key=lambda r: r["score"], reverse=True)
            top_rows = all_rows[: request.top_k]

            chunks = []
            for row in top_rows:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        source_code=row["source_code"] or "",
                        document_id=row["document_id"],
                        document_title=row["name"] or "",
                        url=row["url"] or "",
                        last_modified_date=str(row["last_modified_date"]),
                        metadata={
                            **(json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"] or {}),
                            "embedding_model": model_name,
                            "source": row["source"],
                        },
                        source=row["source"],
                        score=float(row["score"]),
                    )
                )

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
                source_searched=",".join(request.sources) if request.sources else "all",
            )

        except Exception as e:
            logger.error(
                "Error during vector search",
                extra={"error": str(e), "sources": request.sources},
                exc_info=True,
            )
            raise

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
        """Find documents where ALL concepts exist across any of their chunks.

        Uses INTERSECT to efficiently narrow qualifying documents: each concept
        produces a set of document_ids via a GIN index scan, and INTERSECT
        keeps only documents present in every set.
        """
        concept_tsqueries = [_term_to_tsquery(c) for c in query_concepts]

        source_predicate = "source = $2" if source is not None else ""
        corpus_predicate = "retrieval_corpus = $3" if source is not None else "retrieval_corpus = $2"
        intersect_parts: list[str] = []
        for tsq in concept_tsqueries:
            where_predicates = [corpus_predicate, f"tsv @@ to_tsquery('english', '{tsq}')"]
            if source_predicate:
                where_predicates.insert(0, source_predicate)
            intersect_parts.append(
                "SELECT document_id FROM document_embeddings "
                f"WHERE {' AND '.join(where_predicates)}"
            )
        qualifying_docs_sql = " INTERSECT ".join(intersect_parts)

        tsquery_parts = [_term_to_tsquery(c) for c in query_concepts]
        tsquery_wrapped = [f"({p})" if " <-> " in p else p for p in tsquery_parts]
        tsquery_or = " | ".join(tsquery_wrapped)

        term_checks = []
        for i, concept in enumerate(query_concepts):
            tsq = _term_to_tsquery(concept)
            term_checks.append(
                f"(de.tsv @@ to_tsquery('english', '{tsq}'))::int AS has_term_{i}"
            )
        term_check_sql = ", ".join(term_checks)
        total_terms = len(query_concepts)
        term_count_expr = " + ".join(f"has_term_{i}" for i in range(total_terms))

        logger.info(
            "Executing match_all keyword search (INTERSECT approach)",
            extra={"source": source or "all", "concepts": query_concepts, "top_k": top_k},
        )

        # When source is provided it occupies $2, so top_k and max_chunks shift up by one.
        if source is not None:
            top_k_param = "$4"
            max_chunks_param = "$5"
            chunk_terms_source_filter = "AND de.source = $2"
            chunk_terms_corpus_filter = "AND de.retrieval_corpus = $3"
        else:
            top_k_param = "$3"
            max_chunks_param = "$4"
            chunk_terms_source_filter = ""
            chunk_terms_corpus_filter = "AND de.retrieval_corpus = $2"

        if max_chunks_per_document is not None:
            tail_sql = f""",
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
        else:
            tail_sql = f"""
                SELECT chunk_id, text, source_code, document_id, name, url, metadata,
                       source, last_modified_date, score, term_count, base_score
                FROM scored_chunks
                ORDER BY score DESC
                LIMIT {top_k_param};"""

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

        if source is not None:
            if max_chunks_per_document is not None:
                return list(
                    await conn.fetch(
                        sql,
                        tsquery_or,
                        source,
                        retrieval_corpus,
                        top_k,
                        max_chunks_per_document,
                    )
                )
            return list(await conn.fetch(sql, tsquery_or, source, retrieval_corpus, top_k))
        else:
            if max_chunks_per_document is not None:
                return list(await conn.fetch(sql, tsquery_or, retrieval_corpus, top_k, max_chunks_per_document))
            return list(await conn.fetch(sql, tsquery_or, retrieval_corpus, top_k))

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

            async with pool.acquire() as conn:
                search_start = time.time()

                sources_to_search = request.sources or []
                retrieval_corpus = request.retrieval_corpus

                if sources_to_search:
                    for source in sources_to_search:
                        if request.match_all:
                            rows = await self._keyword_search_match_all(
                                conn=conn,
                                source=source,
                                retrieval_corpus=retrieval_corpus,
                                query_concepts=terms,
                                top_k=request.top_k,
                                max_chunks_per_document=request.max_chunks_per_document,
                            )
                            all_rows.extend([dict(r) for r in rows])
                        else:
                            tsquery_str = " | ".join(terms)
                            logger.info(
                                "Executing keyword search (OR logic)",
                                extra={"source": source, "tsquery": tsquery_str},
                            )

                            if request.max_chunks_per_document is not None:
                                sql = """
                                    WITH scored_chunks AS (
                                        SELECT
                                            de.chunk_id, de.text, de.source_code, de.document_id,
                                            dm.name, dm.url,
                                            (de.metadata)::jsonb AS metadata,
                                            de.source, dm.last_modified_date,
                                            ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                                        FROM document_embeddings de
                                        LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                        WHERE de.source = $2
                                          AND de.retrieval_corpus = $5
                                          AND de.tsv IS NOT NULL
                                          AND de.tsv @@ to_tsquery('english', $1)
                                    ),
                                    ranked_chunks AS (
                                        SELECT *,
                                            ROW_NUMBER() OVER (
                                                PARTITION BY document_id ORDER BY score DESC
                                            ) AS doc_rank
                                        FROM scored_chunks
                                    )
                                    SELECT chunk_id, text, source_code, document_id, name, url,
                                           metadata, source, last_modified_date, score
                                    FROM ranked_chunks
                                    WHERE doc_rank <= $4
                                    ORDER BY score DESC
                                    LIMIT $3;
                                """
                                rows = await conn.fetch(
                                    sql, tsquery_str, source,
                                    request.top_k, request.max_chunks_per_document,
                                    retrieval_corpus,
                                )
                            else:
                                sql = """
                                    SELECT
                                        de.chunk_id, de.text, de.source_code, de.document_id,
                                        dm.name, dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source, dm.last_modified_date,
                                        ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                                    FROM document_embeddings de
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    WHERE de.source = $2
                                      AND de.retrieval_corpus = $4
                                      AND de.tsv IS NOT NULL
                                      AND de.tsv @@ to_tsquery('english', $1)
                                    ORDER BY score DESC
                                    LIMIT $3;
                                """
                                rows = await conn.fetch(sql, tsquery_str, source, request.top_k, retrieval_corpus)

                            all_rows.extend([dict(r) for r in rows])

                        logger.debug(f"Source '{source}' returned {len(rows)} results")
                else:
                    # Full corpus keyword search — no source filter
                    if request.match_all:
                        rows = await self._keyword_search_match_all(
                            conn=conn,
                            source=None,
                            retrieval_corpus=retrieval_corpus,
                            query_concepts=terms,
                            top_k=request.top_k,
                            max_chunks_per_document=request.max_chunks_per_document,
                        )
                        all_rows.extend([dict(r) for r in rows])
                    else:
                        tsquery_str = " | ".join(terms)
                        logger.info("Executing keyword search (OR logic, all sources)", extra={"tsquery": tsquery_str})

                        if request.max_chunks_per_document is not None:
                            sql = """
                                WITH scored_chunks AS (
                                    SELECT
                                        de.chunk_id, de.text, de.source_code, de.document_id,
                                        dm.name, dm.url,
                                        (de.metadata)::jsonb AS metadata,
                                        de.source, dm.last_modified_date,
                                        ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                                    FROM document_embeddings de
                                    LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                    WHERE de.retrieval_corpus = $4
                                      AND de.tsv IS NOT NULL
                                      AND de.tsv @@ to_tsquery('english', $1)
                                ),
                                ranked_chunks AS (
                                    SELECT *,
                                        ROW_NUMBER() OVER (
                                            PARTITION BY document_id ORDER BY score DESC
                                        ) AS doc_rank
                                    FROM scored_chunks
                                )
                                SELECT chunk_id, text, source_code, document_id, name, url,
                                       metadata, source, last_modified_date, score
                                FROM ranked_chunks
                                WHERE doc_rank <= $3
                                ORDER BY score DESC
                                LIMIT $2;
                            """
                            rows = await conn.fetch(
                                sql,
                                tsquery_str,
                                request.top_k,
                                request.max_chunks_per_document,
                                retrieval_corpus,
                            )
                        else:
                            sql = """
                                SELECT
                                    de.chunk_id, de.text, de.source_code, de.document_id,
                                    dm.name, dm.url,
                                    (de.metadata)::jsonb AS metadata,
                                    de.source, dm.last_modified_date,
                                    ts_rank_cd(de.tsv, to_tsquery('english', $1), 32) AS score
                                FROM document_embeddings de
                                LEFT JOIN document_metadata dm ON de.document_id = dm.document_id AND de.retrieval_corpus = dm.retrieval_corpus
                                WHERE de.retrieval_corpus = $3
                                  AND de.tsv IS NOT NULL
                                  AND de.tsv @@ to_tsquery('english', $1)
                                ORDER BY score DESC
                                LIMIT $2;
                            """
                            rows = await conn.fetch(sql, tsquery_str, request.top_k, retrieval_corpus)

                        all_rows.extend([dict(r) for r in rows])
                    logger.debug(f"Full corpus keyword search returned {len(all_rows)} results")

                search_duration = (time.time() - search_start) * 1000

            all_rows.sort(key=lambda r: r["score"], reverse=True)
            top_rows = all_rows[: request.top_k]

            chunks = []
            for row in top_rows:
                chunk_metadata: dict = {
                    **(json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"] or {}),
                    "search_type": "keyword",
                    "keyword_ranker": "fts_ts_rank_cd",
                    "source": row["source"],
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

                chunks.append(
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        source_code=row.get("source_code") or "",
                        document_id=row["document_id"],
                        document_title=row["name"] or "",
                        url=row.get("url") or "",
                        last_modified_date=str(row["last_modified_date"]),
                        metadata=chunk_metadata,
                        source=row["source"],
                        score=float(row["score"]),
                    )
                )

            total_duration = (time.time() - start_time) * 1000

            logger.info(
                "Keyword search completed",
                extra={
                    "search": {
                        "results_count": len(chunks),
                        "unique_documents": len(set(c.document_id for c in chunks)),
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
                source_searched=",".join(request.sources) if request.sources else "all",
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
                source_searched=",".join(request.sources) if request.sources else "all",
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
            source_searched=",".join(request.sources) if request.sources else "all",
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