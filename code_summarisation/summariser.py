import asyncio
import logging
import math
import os
import re
from typing import Optional

from anthropic import APIStatusError, APITimeoutError, AsyncAnthropic

from prompts import (
    format_chunk_prompt,
    format_migration_prompt,
    format_repo_summary_prompt,
    format_summary_prompt,
)

log = logging.getLogger(__name__)

CORE_REQUIRED_HEADINGS = [
    "## Purpose",
    "## Key Business Logic",
]


def add_line_numbers(content: str) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(content.splitlines()))


class Summariser:
    def __init__(self) -> None:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.getenv("OPENAI_BASE_URL")
        self.default_model = os.getenv("OPENAI_MODEL", "claude-sonnet-4-6")

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    async def summarise(
        self,
        *,
        repo: str,
        file_path: str,
        content: str,
        language: str = "csharp",
        model: Optional[str] = None,
        tenant_context: str,
    ) -> tuple[str, str, int, int]:
        """Return (summary_markdown, model_used, input_tokens, output_tokens)."""
        model_id = model or self.default_model
        prompt = format_summary_prompt(
            repo=repo,
            file_path=file_path,
            language=language,
            content=add_line_numbers(content),
            tenant_context=tenant_context,
        )

        max_tokens = int(os.getenv("SUMMARISER_MAX_TOKENS", "4096"))
        retry_max_tokens = int(os.getenv("SUMMARISER_RETRY_MAX_TOKENS", "8192"))

        summary, in_tok, out_tok = await self._generate_summary(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        issue = self._validate_summary(summary)
        if issue:
            retry_summary, retry_in_tok, retry_out_tok = await self._generate_summary(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max(retry_max_tokens, max_tokens),
            )
            summary = retry_summary
            in_tok += retry_in_tok
            out_tok += retry_out_tok

            retry_issue = self._validate_summary(summary)
            if retry_issue:
                raise ValueError(
                    f"Summary validation failed after retry for {file_path}: {retry_issue}"
                )

        return summary, model_id, in_tok, out_tok

    async def summarise_migrations(
        self,
        *,
        repo: str,
        files: list[tuple[str, str]],
        model: Optional[str] = None,
        tenant_context: str,
    ) -> tuple[str, str, int, int]:
        """Aggregate ordered SQL migration files into a current-state schema summary.

        Args:
            files: List of (file_path, content) tuples, ordered oldest→newest.

        Returns:
            (summary_markdown, model_used, input_tokens, output_tokens)
        """
        model_id = model or self.default_model

        parts = []
        for file_path, content in files:
            numbered = add_line_numbers(content)
            parts.append(f"-- FILE: {file_path}\n{numbered}")
        aggregate_content = "\n\n".join(parts)

        prompt = format_migration_prompt(
            repo=repo, content=aggregate_content, tenant_context=tenant_context
        )

        max_tokens = int(os.getenv("SUMMARISER_MAX_TOKENS", "4096"))
        retry_max_tokens = int(os.getenv("SUMMARISER_RETRY_MAX_TOKENS", "8192"))

        summary, in_tok, out_tok = await self._generate_summary(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        if "## Schema Overview" not in summary:
            retry_summary, retry_in_tok, retry_out_tok = await self._generate_summary(
                model_id=model_id,
                prompt=prompt,
                max_tokens=max(retry_max_tokens, max_tokens),
            )
            summary = retry_summary
            in_tok += retry_in_tok
            out_tok += retry_out_tok

            if "## Schema Overview" not in summary:
                raise ValueError(
                    f"Migration summary validation failed after retry for {repo}: missing ## Schema Overview"
                )

        return summary, model_id, in_tok, out_tok

    async def generate_repo_summary(
        self,
        *,
        repo: str,
        summaries: dict[str, str],
        tenant_context: str,
        model: Optional[str] = None,
        max_files_per_chunk: int = 25,
    ) -> tuple[str, str, int, int]:
        """Roll all per-file summaries up into one repo-level architectural index.

        Stage 1 summarises equal chunks of files concurrently; stage 2 synthesises those
        chunk summaries into the final index, continuing generation on truncation.
        Returns (index_markdown, model_used, input_tokens, output_tokens).
        """
        model_id = model or self.default_model
        if not summaries:
            return "", model_id, 0, 0

        sorted_files = sorted(summaries.items())
        total = len(sorted_files)
        num_chunks = max(1, math.ceil(total / max_files_per_chunk))
        chunk_size = math.ceil(total / num_chunks)
        chunks = [sorted_files[i : i + chunk_size] for i in range(0, total, chunk_size)]

        sem = asyncio.Semaphore(max(1, int(os.getenv("SUMMARISER_INDEX_CONCURRENCY", "5"))))

        async def _guarded(index: int, chunk_files: list) -> tuple[int, str, int, int]:
            async with sem:
                text, cin, cout = await self._summarise_chunk(
                    chunk_index=index,
                    chunk_files=chunk_files,
                    num_chunks=len(chunks),
                    repo=repo,
                    model_id=model_id,
                    tenant_context=tenant_context,
                )
            return index, text, cin, cout

        results = await asyncio.gather(
            *[_guarded(i, chunk) for i, chunk in enumerate(chunks)]
        )

        in_tok = sum(cin for _, _, cin, _ in results)
        out_tok = sum(cout for _, _, _, cout in results)
        chunk_summaries = [(i, text) for i, text, _, _ in sorted(results) if text]

        if not chunk_summaries:
            raise ValueError(f"All index chunks failed for {repo}")

        combined = "\n\n".join(
            f"### Part {i + 1}\n\n{text}" for i, text in chunk_summaries
        )
        final_prompt = format_repo_summary_prompt(
            repo=repo, summaries=combined, tenant_context=tenant_context
        )

        index_md, s2_in, s2_out = await self._synthesise_index(
            repo=repo, model_id=model_id, prompt=final_prompt
        )
        in_tok += s2_in
        out_tok += s2_out

        if "## Overview" not in index_md:
            log.warning("Repo index missing '## Overview'. repo=%s", repo)

        return index_md, model_id, in_tok, out_tok

    async def _summarise_chunk(
        self,
        *,
        chunk_index: int,
        chunk_files: list[tuple[str, str]],
        num_chunks: int,
        repo: str,
        model_id: str,
        tenant_context: str,
    ) -> tuple[str, int, int]:
        """Summarise one chunk of per-file summaries. Returns (text, in_tok, out_tok)."""
        combined = "\n\n".join(
            f"#### {path}\n\n{summary}" for path, summary in chunk_files
        )
        prompt = format_chunk_prompt(
            repo=repo,
            group=f"chunk {chunk_index + 1} of {num_chunks}",
            summaries=combined,
            tenant_context=tenant_context,
        )
        max_tokens = int(os.getenv("SUMMARISER_MAX_TOKENS", "4096"))
        try:
            return await self._generate_summary(
                model_id=model_id, prompt=prompt, max_tokens=max_tokens
            )
        except Exception as exc:
            log.warning(
                "Repo index chunk failed. repo=%s chunk=%d error=%s",
                repo,
                chunk_index,
                exc,
            )
            return "", 0, 0

    async def _synthesise_index(
        self,
        *,
        repo: str,
        model_id: str,
        prompt: str,
        max_continuations: int = 3,
    ) -> tuple[str, int, int]:
        """Stage-2 synthesis; continue generation when truncated on the token limit."""
        max_tokens = int(
            os.getenv(
                "SUMMARISER_INDEX_MAX_TOKENS",
                os.getenv("SUMMARISER_RETRY_MAX_TOKENS", "8192"),
            )
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        parts: list[str] = []
        in_tok = 0
        out_tok = 0

        for attempt in range(max_continuations + 1):
            response = await self._client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=messages,
            )
            text = response.content[0].text if response.content else ""
            if text:
                parts.append(text)
            usage = response.usage
            if usage:
                in_tok += usage.input_tokens
                out_tok += usage.output_tokens

            if getattr(response, "stop_reason", None) != "max_tokens":
                break
            if attempt == max_continuations:
                log.warning("Repo index truncated after retries. repo=%s", repo)
                break

            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue exactly from where you stopped. "
                        "Do not restart or repeat prior content."
                    ),
                }
            )

        return self._sanitize_summary("".join(parts)), in_tok, out_tok

    async def _generate_summary(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Generate one summary attempt and return content and token usage."""
        try:
            response = await self._client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=0.1,  # low temperature for factual summaries
                messages=[{"role": "user", "content": prompt}],
            )
        except APIStatusError as exc:
            if exc.status_code == 429:
                log.warning("LLM rate limit hit. model=%s", model_id)
            else:
                log.error("LLM API error. model=%s status=%s", model_id, exc.status_code)
            raise
        except APITimeoutError:
            log.warning("LLM request timed out. model=%s", model_id)
            raise

        summary = response.content[0].text if response.content else ""
        summary = self._sanitize_summary(summary)
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        return summary, input_tokens, output_tokens

    def _validate_summary(self, summary: str) -> Optional[str]:
        """Return None when summary is structurally complete, else a failure reason."""
        text = summary.strip()
        if not text:
            return "summary is empty"

        missing = [heading for heading in CORE_REQUIRED_HEADINGS if heading not in text]
        if missing:
            return f"missing required heading(s): {', '.join(missing)}"

        # Guard against truncated outputs that end on a section heading.
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not non_empty_lines:
            return "summary contains no content"

        last_line = non_empty_lines[-1]
        if re.match(r"^##\s+", last_line):
            return f"summary ends with heading: {last_line}"

        return None

    def _sanitize_summary(self, summary: str) -> str:
        """Remove model meta-reasoning sections that should not be indexed."""
        text = summary.strip()
        # Drop markdown heading variants such as "## Reasoning" or "### Pre-Summary Reasoning".
        text = re.sub(
            r"^#{2,3}\s*[^\n]*reasoning[^\n]*\n.*?(?=^##\s|\Z)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        ).strip()
        # Drop bold-label style reasoning blocks like "**Reasoning:**" at the top.
        text = re.sub(
            r"^\*\*[^\n]*reasoning[^\n]*\*\*:?\n.*?(?=^##\s|\Z)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        ).strip()
        return text

    async def close(self) -> None:
        await self._client.close()
