import asyncio
import time
from typing import List

from retrieval.clients.search_factory import SearchClientFactory
from retrieval.graph.orchestrator import GraphClient
from retrieval.models.models import SearchRequest, SearchResponse
from retrieval.strategies.rrf import rrf_merge
from retrieval.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridSearchEndpoint:
    def __init__(self, search_client=None):
        self.search_client = search_client or SearchClientFactory.create_search_client()
        self.graph_client = GraphClient(self.search_client)

    async def run(self, request: SearchRequest) -> List[SearchResponse]:
        start = time.time()

        vec_result, kw_result, graph_result = await asyncio.gather(
            self.search_client.search(request),
            self.search_client.search_keyword(request),
            self.graph_client.search(request),
            return_exceptions=True,
        )

        if isinstance(vec_result, Exception):
            raise vec_result
        if isinstance(kw_result, Exception):
            raise kw_result

        vec_response = vec_result
        kw_response = kw_result
        graph_error: Exception | None = None
        if isinstance(graph_result, Exception):
            graph_error = graph_result
            logger.error(
                "Graph retrieval failed during hybrid search; falling back to vector+keyword merge",
                extra={"error": {"type": type(graph_result).__name__, "message": str(graph_result)}},
                exc_info=True,
            )
            graph_response = SearchResponse(
                chunks=[],
                total_results=0,
                search_duration_ms=0.0,
                embedding_duration_ms=0.0,
                model_used="graph-traversal-v1",
                source_searched=",".join(request.sources) if request.sources else "all",
            )
        else:
            graph_response = graph_result

        merged_chunks = rrf_merge([vec_response, kw_response, graph_response], request.top_k)

        if graph_error is not None:
            for chunk in merged_chunks:
                metadata = dict(chunk.metadata or {})
                metadata["graph_error"] = {
                    "type": type(graph_error).__name__,
                    "message": str(graph_error),
                }
                metadata["graph_included"] = False
                chunk.metadata = metadata

        return [SearchResponse(
            chunks=merged_chunks,
            total_results=len(merged_chunks),
            search_duration_ms=(time.time() - start) * 1000,
            embedding_duration_ms=vec_response.embedding_duration_ms,
            model_used=f"{vec_response.model_used}+keyword+graph",
            source_searched=vec_response.source_searched,
        )]
