from typing import List

from retrieval.clients.search_factory import SearchClientFactory
from retrieval.models.models import SearchRequest, SearchResponse


class VectorSearchEndpoint:
    def __init__(self, search_client=None):
        self.search_client = search_client or SearchClientFactory.create_search_client()

    async def run(self, input_data: SearchRequest) -> List[SearchResponse]:
        result = await self.search_client.search(request=input_data)
        return [result]