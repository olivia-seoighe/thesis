import os
import re
from typing import Optional

from openai import AsyncOpenAI

from prompts import format_migration_prompt, format_summary_prompt

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
        self._client = AsyncOpenAI(**kwargs)

    async def summarise(
        self,
        *,
        repo: str,
        file_path: str,
        content: str,
        language: str = "csharp",
        model: Optional[str] = None,
    ) -> tuple[str, str, int, int]:
        """Return (summary_markdown, model_used, input_tokens, output_tokens)."""
        model_id = model or self.default_model
        prompt = format_summary_prompt(
            repo=repo,
            file_path=file_path,
            language=language,
            content=add_line_numbers(content),
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

        prompt = format_migration_prompt(repo=repo, content=aggregate_content)

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

    async def _generate_summary(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Generate one summary attempt and return content and token usage."""
        response = await self._client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # low temperature for factual summaries
            max_tokens=max_tokens,
        )

        summary = response.choices[0].message.content or ""
        summary = self._sanitize_summary(summary)
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

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
