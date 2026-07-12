"""OpenAI-compatible embedding client for the indexing pipeline."""

import os
from typing import List, Optional

from openai import AsyncOpenAI

from indexing.models.document import DocumentChunk, EmbeddedDocumentChunk


class EmbeddingClient:
    def __init__(self) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedDocumentChunk]:
        texts = [c.text for c in chunks]
        embeddings = await self.embed_batch(texts)
        return [
            EmbeddedDocumentChunk(chunk=c, embedding=emb, embedding_model=self.model)
            for c, emb in zip(chunks, embeddings)
        ]

    async def close(self) -> None:
        await self._client.close()
