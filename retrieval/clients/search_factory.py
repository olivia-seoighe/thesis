import logging
from typing import Any, Optional

from clients.search_client import SearchClient

logger = logging.getLogger(__name__)


class SearchClientFactory:
    @classmethod
    def create_search_client(
        cls,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 5,
        **pool_kwargs: Any,
    ) -> SearchClient:
        logger.info("Creating SearchClient")
        return SearchClient(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=min_size,
            max_size=max_size,
            **pool_kwargs,
        )