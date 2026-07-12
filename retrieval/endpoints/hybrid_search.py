import asyncio
import time
from typing import List

from clients.search_factory import SearchClientFactory
from models.models import SearchRequest, SearchResponse
from strategies.rrf import rrf_merge


class HybridSearchEndpoint:
    def __init__(self, search_client=None):
        self.search_client = search_client or SearchClientFactory.create_search_client()

    async def run(self, request: SearchRequest) -> List[SearchResponse]:
        start = time.time()

        vec_response, kw_response = await asyncio.gather(
            self.search_client.search(request),
            self.search_client.search_keyword(request),
            # TODO (RQ1): add graph_client.search(request) as a third parallel call
            # then pass graph_response into rrf_merge below
        )

        merged_chunks = rrf_merge([vec_response, kw_response], request.top_k)
        # TODO (RQ1): rrf_merge([vec_response, kw_response, graph_response], request.top_k)

        return [SearchResponse(
            chunks=merged_chunks,
            total_results=len(merged_chunks),
            search_duration_ms=(time.time() - start) * 1000,
            embedding_duration_ms=vec_response.embedding_duration_ms,
            model_used=vec_response.model_used,
            source_searched=vec_response.source_searched,
        )]
