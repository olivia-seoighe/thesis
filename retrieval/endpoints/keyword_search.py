from typing import List

from clients.search_factory import SearchClientFactory
from models.models import SearchRequest, SearchResponse


class KeywordSearchEndpoint:
    def __init__(self, search_client=None):
        self.search_client = search_client or SearchClientFactory.create_search_client()

    async def run(self, input_data: SearchRequest) -> List[SearchResponse]:
        result = await self.search_client.search_keyword(request=input_data)
        return [result]