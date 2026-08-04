import os
from typing import List, Optional

from openai import AsyncOpenAI


class EmbeddingAPIClient:
    """OpenAI embedding client — direct API or enterprise gateway via OPENAI_BASE_URL."""

    def __init__(self) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.getenv("OPENAI_BASE_URL")  # Optional: enterprise gateway
        self.default_model = os.environ["OPENAI_EMBEDDING_MODEL"]
        dim = os.getenv("EMBEDDING_DIM")
        self.dimensions = int(dim) if dim else None

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)

    def _create_kwargs(self, model_name: Optional[str]) -> dict:
        kwargs: dict = {"model": model_name or self.default_model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    async def embed_single(self, text: str, model_name: Optional[str] = None) -> List[float]:
        response = await self._client.embeddings.create(
            input=text, **self._create_kwargs(model_name)
        )
        return response.data[0].embedding

    async def embed_batch(
        self, texts: List[str], model_name: Optional[str] = None
    ) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts, **self._create_kwargs(model_name)
        )
        items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in items]

    async def close(self) -> None:
        await self._client.close()