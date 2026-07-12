"""Ingest GitHub code summaries into the RAG database.

Expected JSON input (set SUMMARIES_FILE env var to the file path):

    {
      "repository": "service_A",
      "files": [
        {
          "file_path": "src/main.cs",
          "url": "https://github.com/org/service_A/blob/main/src/main.cs",
          "last_modified": "2024-01-01T00:00:00Z",
          "source_code": "...",
          "summary": "## Purpose\\n..."
        }
      ]
    }

Run locally:
    SUMMARIES_FILE=./indexing/summaries/summaries.json python indexing/main.py

Run via Docker Compose:
    docker compose --profile ingest up indexing
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# Allow running as `python indexing/main.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing.chunker.document_chunker import DocumentChunker
from indexing.clients.embedding_client import EmbeddingClient
from indexing.connection_manager import ConnectionManager
from indexing.indexer.postgres_indexer import PostgresIndexer
from indexing.models.document import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Handle common UTC suffix used in payloads.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning(f"Invalid last_modified datetime format: {value}")
        return None


def _is_unchanged(last_modified: datetime, last_indexed_at: datetime) -> bool:
    """Safely compare datetimes even if one value is timezone-naive."""
    if last_modified.tzinfo and not last_indexed_at.tzinfo:
        last_indexed_at = last_indexed_at.replace(tzinfo=last_modified.tzinfo)
    elif last_indexed_at.tzinfo and not last_modified.tzinfo:
        last_modified = last_modified.replace(tzinfo=last_indexed_at.tzinfo)
    return last_modified <= last_indexed_at


def split_summary(summary: str) -> tuple[str, str]:
    """Split summary into embeddable text and source references section.

    Returns (embeddable_text, source_refs) where source_refs is the raw
    '## Source References' block. The embeddable text excludes non-retrieval
    sections like '## Reasoning'.
    """
    marker = "## Source References"
    idx = summary.find(marker)
    if idx == -1:
        body = summary
        source_refs = ""
    else:
        body = summary[:idx]
        source_refs = summary[idx:]

    # Exclude internal planning content from embeddings to reduce retrieval noise.
    embeddable_text = re.sub(
        r"^##\s*Reasoning\b.*?(?=^##\s|\Z)",
        "",
        body,
        flags=re.DOTALL | re.MULTILINE,
    ).strip()

    return embeddable_text, source_refs


def build_document(file_entry: dict, source: str) -> Document:
    file_path = file_entry["file_path"]
    doc_id = hashlib.sha256(f"{source}::{file_path}".encode()).hexdigest()
    embeddable_text, source_refs = split_summary(file_entry.get("summary", ""))
    return Document(
        id=doc_id,
        title=file_path,
        text=embeddable_text,
        source=source,
        url=file_entry.get("url", ""),
        source_code=file_entry.get("source_code", ""),
        last_modified_date=file_entry.get("last_modified"),
        source_refs=source_refs,
    )


async def main() -> None:
    summaries_file = os.environ["SUMMARIES_FILE"]
    skip_unchanged = os.getenv("SKIP_UNCHANGED", "false").lower() == "true"

    with open(summaries_file) as f:
        data = json.load(f)

    files = data.get("files", [])
    source = data.get("repository", "sample-service")
    logger.info(f"Loaded {len(files)} files from {summaries_file} (source={source})")

    db = ConnectionManager()
    embedder = EmbeddingClient()
    indexer = PostgresIndexer(db)
    chunker = DocumentChunker()

    try:
        for i, file_entry in enumerate(files, start=1):
            doc = build_document(file_entry, source)
            if not doc.text.strip():
                logger.warning(f"Skipping empty summary: {doc.title}")
                continue

            if skip_unchanged:
                last_indexed_at = await indexer.get_last_indexed_at(doc.id)
                last_modified = _parse_iso_datetime(doc.last_modified_date)
                if last_indexed_at and last_modified and _is_unchanged(last_modified, last_indexed_at):
                    logger.info(f"[{i}/{len(files)}] Skipping unchanged: {doc.title}")
                    continue

            chunks = chunker.chunk(doc)
            embedded_chunks = await embedder.embed_chunks(chunks)
            await indexer.replace_document(doc, embedded_chunks)

            logger.info(f"[{i}/{len(files)}] {doc.title} → {len(chunks)} chunk(s)")

        logger.info("Ingest complete.")
    finally:
        await db.close()
        await embedder.close()


if __name__ == "__main__":
    asyncio.run(main())
