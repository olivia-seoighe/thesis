from typing import List

from retrieval.clients.search_factory import SearchClientFactory
from retrieval.graph.orchestrator import GraphClient
from retrieval.models.models import SearchRequest, SearchResponse


class GraphSearchEndpoint:
    def __init__(self, search_client=None):
        self.search_client = search_client or SearchClientFactory.create_search_client()
        self.graph_client = GraphClient(self.search_client)

    async def run(self, input_data: SearchRequest, *, hop_policy_mode: str = "adaptive") -> List[SearchResponse]:
        result = await self.graph_client.search(
            request=input_data,
            hop_policy_mode=hop_policy_mode,
        )
        return [result]