import asyncio
import time
from pathlib import Path
from typing import List

from retrieval.clients.search_factory import SearchClientFactory
from retrieval.graph.orchestrator import GraphClient
from retrieval.models.models import SearchRequest, SearchResponse
from retrieval.strategies.service_aware import (
    SERVICE_AWARE_BOOST_OVERFETCH_CAP,
    SERVICE_AWARE_BOOST_OVERFETCH_FACTOR,
    MetadataMode,
    ServiceAwarePlanner,
    apply_service_boost,
    load_service_catalogue,
)
from retrieval.strategies.rrf import rrf_merge
from retrieval.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridSearchEndpoint:
    def __init__(self, search_client=None, service_catalogue_path: Path | None = None):
        self.search_client = search_client or SearchClientFactory.create_search_client()
        self.graph_client = GraphClient(self.search_client)
        path = service_catalogue_path or Path(__file__).resolve().parents[2] / "service_acronyms.json"
        self.service_planner = ServiceAwarePlanner(load_service_catalogue(path) if path.exists() else ())

    async def run(
        self,
        request: SearchRequest,
        *,
        service_aware: bool = False,
    ) -> List[SearchResponse]:
        start = time.time()
        decision = None
        if service_aware:
            decision = self.service_planner.plan(request.query)
            if decision.metadata_mode == MetadataMode.HARD_FILTER:
                request = request.model_copy(update={"sources": list(decision.filter_services)})

        final_top_k = request.top_k
        merge_top_k = final_top_k
        if decision is not None and decision.metadata_mode == MetadataMode.BOOST:
            merge_top_k = min(final_top_k * SERVICE_AWARE_BOOST_OVERFETCH_FACTOR, SERVICE_AWARE_BOOST_OVERFETCH_CAP)
            request = request.model_copy(update={"top_k": merge_top_k})

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

        merged_chunks = rrf_merge([vec_response, kw_response, graph_response], merge_top_k)
        if decision is not None and decision.metadata_mode == MetadataMode.BOOST:
            merged_chunks = apply_service_boost(chunks=merged_chunks, boost_services=decision.boost_services)
        merged_chunks = merged_chunks[:final_top_k]

        if graph_error is not None:
            for chunk in merged_chunks:
                metadata = dict(chunk.metadata or {})
                metadata["graph_error"] = {"type": type(graph_error).__name__, "message": str(graph_error)}
                metadata["graph_included"] = False
                chunk.metadata = metadata

        return [SearchResponse(
            chunks=merged_chunks,
            total_results=len(merged_chunks),
            search_duration_ms=(time.time() - start) * 1000,
            embedding_duration_ms=vec_response.embedding_duration_ms,
            model_used=f"{vec_response.model_used}+{kw_response.model_used}+graph",
            source_searched=vec_response.source_searched,
        )]
