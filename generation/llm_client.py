"""LLM generation client — calls the enterprise gateway with RAG context."""

import os
from typing import Optional

from anthropic import AsyncAnthropic

from models import Citation
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _build_context_block(citations: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(citations, 1):
        parts.append(
            f"[Source {i}]: {chunk['document_title']} ({chunk['url']})\n"
            f"Relevance: {chunk['score']:.3f}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


class LLMClient:
    def __init__(self) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.getenv("OPENAI_BASE_URL")
        self.default_model = os.getenv("OPENAI_MODEL", "claude-sonnet-4-6")

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    async def generate(
        self,
        *,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
        model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Return (answer_text, model_id)."""
        model_id = model or self.default_model
        context = _build_context_block(chunks)

        messages: list[dict] = []

        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(context=context, query=query),
        })

        response = await self._client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            temperature=0.1,
            messages=messages,
        )

        answer = response.content[0].text if response.content else ""
        return answer, model_id

    async def close(self) -> None:
        await self._client.close()
