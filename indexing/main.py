"""Ingest GitHub code summaries into the RAG database.

Expected per-repo JSON input (SUMMARIES_DIR globs <dir>/*/summaries.json; SUMMARIES_FILE for one file):

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
    SUMMARIES_DIR=./summaries python indexing/main.py   # ingests every <repo>/summaries.json

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

from dotenv import load_dotenv

load_dotenv()

# Allow running as `python indexing/main.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing.chunker.document_chunker import DocumentChunker
from indexing.clients.embedding_client import EmbeddingClient
from indexing.connection_manager import ConnectionManager
from indexing.graph import GraphExtractor, GraphIndexer
from indexing.indexer.postgres_indexer import PostgresIndexer
from indexing.models.document import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

RETRIEVAL_CORPUS_SUMMARIES = "summaries"
RETRIEVAL_CORPUS_CODE = "code"
_RETRIEVAL_CORPUS_ALIASES = {
    "summary": RETRIEVAL_CORPUS_SUMMARIES,
    "summaries": RETRIEVAL_CORPUS_SUMMARIES,
    "code": RETRIEVAL_CORPUS_CODE,
    "source_code": RETRIEVAL_CORPUS_CODE,
}


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

    # Strip inline line-range citations
    # These stay in the summary body for provenance but shouldn't leak into generated answers
    embeddable_text = re.sub(
        r"\s?\((?:[\w./-]+:)?L\d+(?:[-–—]L?\d+)?\)",
        "",
        embeddable_text,
    )

    return embeddable_text, source_refs


def build_document(file_entry: dict, source: str, retrieval_corpus: str) -> Document:
    file_path = file_entry["file_path"]
    doc_id = hashlib.sha256(f"{source}::{file_path}".encode()).hexdigest()
    embeddable_text, source_refs = split_summary(file_entry.get("summary", ""))
    if retrieval_corpus == RETRIEVAL_CORPUS_CODE:
        index_text = str(file_entry.get("source_code", "") or "")
        source_refs = ""
    else:
        index_text = embeddable_text

    return Document(
        id=doc_id,
        title=file_path,
        text=index_text,
        source=source,
        url=file_entry.get("url", ""),
        source_code=file_entry.get("source_code", ""),
        last_modified_date=file_entry.get("last_modified") or None,
        source_refs=source_refs,
        retrieval_corpus=retrieval_corpus,
    )


def _resolve_retrieval_corpus() -> str:
    raw_value = os.getenv("INDEXING_RETRIEVAL_CORPUS", RETRIEVAL_CORPUS_SUMMARIES)
    token = str(raw_value).strip().lower()
    resolved = _RETRIEVAL_CORPUS_ALIASES.get(token)
    if resolved is None:
        allowed = ", ".join(sorted(_RETRIEVAL_CORPUS_ALIASES))
        raise ValueError(
            f"Invalid INDEXING_RETRIEVAL_CORPUS={raw_value!r}. "
            f"Allowed values: {allowed}."
        )
    return resolved


def _graph_extraction_priority(file_path: str) -> int:
    path = (file_path or "").lower()
    if "asyncapi" in path:
        return 0
    if "schema_state_" in path:
        return 1
    if "configmap" in path:
        return 2
    if path.endswith(".cs"):
        return 3
    if path.endswith("appsettings.prod.json"):
        return 4
    if path.endswith("appsettings.json"):
        return 5
    if "ingress" in path:
        return 6
    return 10


def _get_summary_files() -> list[Path]:
    """Return the summaries.json files to ingest.

    SUMMARIES_DIR (preferred) globs <dir>/*/summaries.json — one per repo.
    SUMMARIES_FILE ingests a single file.
    """
    summaries_dir = os.getenv("SUMMARIES_DIR")
    if summaries_dir:
        paths = sorted(Path(summaries_dir).glob("*/summaries.json"))
        services = os.getenv("SUMMARY_SERVICES")
        if services:
            wanted = {s.strip() for s in services.split(",") if s.strip()}
            paths = [p for p in paths if p.parent.name in wanted]
            logger.info(f"Filtering ingest to services: {sorted(wanted)}")
        if not paths:
            logger.warning(f"No */summaries.json found under {summaries_dir}")
        return paths
    return [Path(os.environ["SUMMARIES_FILE"])]


def _get_source(data: dict, summaries_file: Path) -> str:
    source = str(data.get("repository", "")).strip()
    if not source:
        raise ValueError(
            f"Missing required 'repository' field in summaries file: {summaries_file}"
        )
    return source


async def main() -> None:
    skip_unchanged = os.getenv("SKIP_UNCHANGED", "false").lower() == "true"
    graph_indexing_enabled = os.getenv("GRAPH_INDEXING_ENABLED", "true").lower() == "true"
    graph_only = os.getenv("GRAPH_ONLY", "false").lower() == "true"
    retrieval_corpus = _resolve_retrieval_corpus()
    summary_files = _get_summary_files()

    db = ConnectionManager()
    embedder: EmbeddingClient | None = None
    indexer: PostgresIndexer | None = None
    chunker: DocumentChunker | None = None
    if not graph_only:
        embedder = EmbeddingClient()
        indexer = PostgresIndexer(db)
        chunk_size = int(os.getenv("CHUNK_SIZE", "1500"))
        overlap_ratio = float(os.getenv("CHUNK_OVERLAP", "0.15"))
        chunker = DocumentChunker(chunk_size=chunk_size, overlap_ratio=overlap_ratio)
        logger.info(f"Chunker: chunk_size={chunk_size} words, overlap_ratio={overlap_ratio}")
    elif not graph_indexing_enabled:
        raise ValueError("GRAPH_ONLY=true requires GRAPH_INDEXING_ENABLED=true")

    graph_extractor = GraphExtractor()
    graph_indexer = GraphIndexer(db)
    logger.info(f"Graph-only mode: {graph_only}")
    logger.info(f"Graph indexing enabled: {graph_indexing_enabled}")
    logger.info(f"Vector retrieval corpus mode: {retrieval_corpus}")

    try:
        for summaries_file in summary_files:
            with open(summaries_file) as f:
                data = json.load(f)

            files = data.get("files", [])
            files = sorted(
                files,
                key=lambda file_entry: _graph_extraction_priority(
                    str(file_entry.get("file_path", ""))
                ),
            )
            source = _get_source(data, summaries_file)
            logger.info(f"Loaded {len(files)} files from {summaries_file} (source={source})")
            graph_extractor.register_repo_files(
                source,
                [str(file_entry.get("file_path", "")) for file_entry in files],
            )
            if graph_indexing_enabled:
                await graph_indexer.upsert_service_node(source)

            for i, file_entry in enumerate(files, start=1):
                doc = build_document(file_entry, source, retrieval_corpus)

                if graph_indexing_enabled:
                    source_path = str(file_entry.get("file_path", ""))
                    triples = graph_extractor.extract(
                        summary=file_entry.get("summary", ""),
                        document_title=source_path,
                        service=source,
                        source_code=file_entry.get("source_code", ""),
                    )
                    for triple in triples:
                        triple.properties.setdefault("document_id", doc.id)
                    await graph_indexer.replace_source_triples(
                        source_repo=source,
                        source_path=source_path,
                        triples=triples,
                    )
                    logger.info(
                        f"[{i}/{len(files)}] {doc.title} → replaced graph evidence ({len(triples)} triple(s))"
                    )
                    if triples:
                        logger.debug(
                            f"[{i}/{len(files)}] {doc.title} → {len(triples)} graph triple(s)"
                        )

                if graph_only:
                    continue

                if not doc.text.strip():
                    logger.warning(f"Skipping empty {retrieval_corpus} text: {doc.title}")
                    continue

                if skip_unchanged:
                    if indexer is None:
                        raise RuntimeError("Vector indexer is not initialized")
                    last_indexed_at = await indexer.get_last_indexed_at(doc.id, doc.retrieval_corpus)
                    last_modified = _parse_iso_datetime(doc.last_modified_date)
                    if last_indexed_at and last_modified and _is_unchanged(last_modified, last_indexed_at):
                        logger.info(f"[{i}/{len(files)}] Skipping unchanged: {doc.title}")
                        continue

                if chunker is None or embedder is None or indexer is None:
                    raise RuntimeError("Vector indexing dependencies are not initialized")
                chunks = chunker.chunk(doc)
                embedded_chunks = await embedder.embed_chunks(chunks)
                await indexer.replace_document(doc, embedded_chunks)

                logger.info(f"[{i}/{len(files)}] {doc.title} → {len(chunks)} chunk(s)")

        logger.info("Ingest complete.")
    finally:
        await db.close()
        if embedder is not None:
            await embedder.close()


if __name__ == "__main__":
    asyncio.run(main())
