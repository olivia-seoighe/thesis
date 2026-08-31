import os
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, Request

from retrieval.endpoints.graph_search import GraphSearchEndpoint
from retrieval.endpoints.hybrid_search import HybridSearchEndpoint
from retrieval.endpoints.keyword_search import KeywordSearchEndpoint
from retrieval.endpoints.vector_search import VectorSearchEndpoint
from retrieval.models.models import SearchRequest, SearchResponse
from retrieval.utils.logging_config import get_logger, get_request_id

load_dotenv()

BUILD_TIMESTAMP = os.getenv("BUILD_TIMESTAMP", "unknown")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
QUERY_PARAM_DESCRIPTION = "The query to search for"
TOP_K_PARAM_DESCRIPTION = "Number of results to return"
SOURCE_PARAM_DESCRIPTION = (
    "Source filter(s). Single source or comma-separated list. Omit to search all sources."
)
CORPUS_PARAM_DESCRIPTION = "Retrieval corpus selector: summaries (default), code, or all."

vector_search_endpoint = VectorSearchEndpoint()
keyword_search_endpoint = KeywordSearchEndpoint(
    search_client=vector_search_endpoint.search_client
)
graph_search_endpoint = GraphSearchEndpoint(
    search_client=vector_search_endpoint.search_client
)
hybrid_search_endpoint = HybridSearchEndpoint(
    search_client=vector_search_endpoint.search_client
)
logger = get_logger(__name__)

router = APIRouter()


def _parse_sources_param(source: str | None) -> list[str]:
    return [value.strip() for value in source.split(",")] if source else []


def _normalize_corpus_param(corpus: str | None) -> str:
    if not isinstance(corpus, str):
        corpus = None
    token = (corpus or "summaries").strip().lower()
    if token in {"summary", "summaries"}:
        return "summaries"
    if token in {"code", "source_code"}:
        return "code"
    if token == "all":
        return "all"
    raise HTTPException(status_code=400, detail=f"Invalid corpus value: {corpus!r}")


def _results_count(results: list[SearchResponse]) -> int:
    return sum(len(item.chunks) for item in results)


@router.get("/search/vector", response_model=List[SearchResponse])
async def search_vector(
    request: Request,
    query: str = Query(..., description=QUERY_PARAM_DESCRIPTION),
    top_k: int = Query(10, description=TOP_K_PARAM_DESCRIPTION),
    source: str | None = Query(
        None,
        description=SOURCE_PARAM_DESCRIPTION,
    ),
    entity_filter: str | None = Query(
        None,
        description="Optional entity filter to narrow search to documents containing this entity.",
    ),
    corpus: str = Query("summaries", description=CORPUS_PARAM_DESCRIPTION),
):
    logger.info(
        "Processing vector search request",
        extra={
            "search": {
                "query": query,
                "top_k": top_k,
                "source": source,
                "entity_filter": entity_filter,
                "corpus": corpus,
                "query_length": len(query),
            }
        },
    )
    try:
        sources = _parse_sources_param(source)
        retrieval_corpus = _normalize_corpus_param(corpus)
        search_request = SearchRequest(
            query=query,
            top_k=top_k,
            sources=sources,
            entity_filter=entity_filter,
            retrieval_corpus=retrieval_corpus,
        )
        results = await vector_search_endpoint.run(search_request)
        logger.info(
            "Search completed",
            extra={"search": {"results_count": _results_count(results)}},
        )
        return results
    except Exception as e:
        logger.error(
            "Search failed",
            extra={"error": {"type": type(e).__name__, "message": str(e)}},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/keyword", response_model=List[SearchResponse])
async def search_keyword(
    request: Request,
    query: str = Query(..., description=QUERY_PARAM_DESCRIPTION),
    top_k: int = Query(10, description=TOP_K_PARAM_DESCRIPTION),
    source: str | None = Query(
        None,
        description=SOURCE_PARAM_DESCRIPTION,
    ),
    match_all: bool = Query(
        False,
        description="If true, all query terms must exist somewhere in the document.",
    ),
    max_chunks_per_document: int | None = Query(
        3,
        description="Maximum chunks per document. Default 3. None = no limit.",
    ),
    keyword_ranker: Literal["fts", "bm25"] = Query(
        "fts",
        description="Keyword ranker: fts (default) or bm25.",
    ),
    corpus: str = Query("summaries", description=CORPUS_PARAM_DESCRIPTION),
):
    logger.info(
        "Processing keyword search request",
        extra={
            "search": {
                "query": query,
                "top_k": top_k,
                "source": source,
                "match_all": match_all,
                "max_chunks_per_document": max_chunks_per_document,
                "keyword_ranker": keyword_ranker,
                "corpus": corpus,
                "query_length": len(query),
            }
        },
    )
    try:
        sources = _parse_sources_param(source)
        retrieval_corpus = _normalize_corpus_param(corpus)
        search_request = SearchRequest(
            query=query,
            top_k=top_k,
            sources=sources,
            match_all=match_all,
            max_chunks_per_document=max_chunks_per_document,
            keyword_ranker=keyword_ranker,
            retrieval_corpus=retrieval_corpus,
        )
        results = await keyword_search_endpoint.run(search_request)
        logger.info(
            "Search completed",
            extra={"search": {"results_count": _results_count(results)}},
        )
        return results
    except Exception as e:
        logger.error(
            "Search failed",
            extra={"error": {"type": type(e).__name__, "message": str(e)}},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/hybrid", response_model=List[SearchResponse])
async def search_hybrid(
    request: Request,
    query: str = Query(..., description=QUERY_PARAM_DESCRIPTION),
    top_k: int = Query(10, description=TOP_K_PARAM_DESCRIPTION),
    source: str | None = Query(
        None,
        description=SOURCE_PARAM_DESCRIPTION,
    ),
    keyword_ranker: Literal["fts", "bm25"] = Query(
        "fts",
        description="Keyword ranker used by hybrid endpoint: fts (default) or bm25.",
    ),
    corpus: str = Query("summaries", description=CORPUS_PARAM_DESCRIPTION),
):
    logger.info(
        "Processing hybrid search request",
        extra={
            "search": {
                "query": query,
                "top_k": top_k,
                "source": source,
                "keyword_ranker": keyword_ranker,
                "corpus": corpus,
            }
        },
    )
    try:
        sources = _parse_sources_param(source)
        retrieval_corpus = _normalize_corpus_param(corpus)
        search_request = SearchRequest(
            query=query,
            top_k=top_k,
            sources=sources,
            keyword_ranker=keyword_ranker,
            retrieval_corpus=retrieval_corpus,
        )
        results = await hybrid_search_endpoint.run(search_request)
        logger.info(
            "Hybrid search completed",
            extra={"search": {"results_count": _results_count(results)}},
        )
        return results
    except Exception as e:
        logger.error(
            "Hybrid search failed",
            extra={"error": {"type": type(e).__name__, "message": str(e)}},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/graph", response_model=List[SearchResponse])
async def search_graph(
    request: Request,
    query: str = Query(..., description=QUERY_PARAM_DESCRIPTION),
    top_k: int = Query(10, description=TOP_K_PARAM_DESCRIPTION),
    source: str | None = Query(
        None,
        description=SOURCE_PARAM_DESCRIPTION,
    ),
    hop_policy: str = Query(
        "adaptive",
        description="Graph hop policy mode: adaptive (default) or fixed.",
    ),
    corpus: str = Query("summaries", description=CORPUS_PARAM_DESCRIPTION),
):
    logger.info(
        "Processing graph search request",
        extra={
            "search": {
                "query": query,
                "top_k": top_k,
                "source": source,
                "hop_policy": hop_policy,
                "corpus": corpus,
            }
        },
    )
    try:
        sources = _parse_sources_param(source)
        retrieval_corpus = _normalize_corpus_param(corpus)
        search_request = SearchRequest(
            query=query,
            top_k=top_k,
            sources=sources,
            retrieval_corpus=retrieval_corpus,
        )
        results = await graph_search_endpoint.run(
            search_request,
            hop_policy_mode=hop_policy,
        )
        logger.info(
            "Graph search completed",
            extra={"search": {"results_count": _results_count(results)}},
        )
        return results
    except Exception as e:
        logger.error(
            "Graph search failed",
            extra={"error": {"type": type(e).__name__, "message": str(e)}},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def list_sources(corpus: str = Query("all", description=CORPUS_PARAM_DESCRIPTION)) -> dict:
    """Return the distinct indexed sources (repositories) for the UI selector."""
    retrieval_corpus = _normalize_corpus_param(corpus)
    if retrieval_corpus == "all":
        rows = await vector_search_endpoint.search_client.fetch(
            "SELECT DISTINCT source FROM document_embeddings WHERE source IS NOT NULL ORDER BY source"
        )
    else:
        rows = await vector_search_endpoint.search_client.fetch(
            """
            SELECT DISTINCT source
            FROM document_embeddings
            WHERE source IS NOT NULL
              AND COALESCE(metadata->>'retrieval_corpus', 'summaries') = $1
            ORDER BY source
            """,
            retrieval_corpus,
        )
    return {"sources": [r["source"] for r in rows]}


@router.get("/version")
async def version() -> dict:
    return {
        "build_timestamp": BUILD_TIMESTAMP,
        "git_commit": GIT_COMMIT,
        "features": {
            "entity_filter": True,
            "cross_chunk_match_all": True,
            "document_diversity": True,
            "graph_search": True,
            "keyword_ranker": ["fts", "bm25"],
        },
    }


@router.get("/live")
async def live():
    logger.debug("Liveness check requested")
    return {"status": "alive", "request_id": get_request_id()}


@router.get("/ready")
async def ready():
    logger.debug("Readiness check requested")
    try:
        is_healthy = await vector_search_endpoint.search_client.health_check()
        if not is_healthy:
            logger.error("Search backend health check failed")
            raise HTTPException(status_code=503, detail="Search backend unhealthy")
        return {"status": "ready", "request_id": get_request_id()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Search backend error during readiness check",
            extra={"error": {"type": type(e).__name__, "message": str(e)}},
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail=f"Search backend error: {str(e)}")
