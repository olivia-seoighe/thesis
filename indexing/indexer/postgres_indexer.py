import json
import logging
from datetime import datetime

from indexing.connection_manager import ConnectionManager
from indexing.models import Document, EmbeddedDocumentChunk

log = logging.getLogger(__name__)

INSERT_CHUNK_SQL = """
    INSERT INTO document_embeddings
        (chunk_id, text, source_code, document_id, document_title, embedding_3072, tsv, metadata, source)
    VALUES (
        $1, $2, $3, $4, $5,
        $6::halfvec,
        setweight(to_tsvector('english', coalesce($2, '')), 'A') ||
        setweight(to_tsvector('english', coalesce($5, '')), 'B'),
        $7::jsonb, $8
    )
    ON CONFLICT (chunk_id) DO UPDATE SET
        text            = EXCLUDED.text,
        source_code     = EXCLUDED.source_code,
        document_id     = EXCLUDED.document_id,
        document_title  = EXCLUDED.document_title,
        embedding_3072  = EXCLUDED.embedding_3072,
        tsv             = EXCLUDED.tsv,
        metadata        = EXCLUDED.metadata,
        source          = EXCLUDED.source
"""

UPSERT_METADATA_SQL = """
    INSERT INTO document_metadata (document_id, name, url, source, last_modified_date, source_refs)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (document_id) DO UPDATE SET
        name               = EXCLUDED.name,
        url                = EXCLUDED.url,
        last_modified_date = EXCLUDED.last_modified_date,
        source_refs        = EXCLUDED.source_refs,
        last_indexed_at    = NOW()
"""


class PostgresIndexer:
    def __init__(self, connection_manager: ConnectionManager):
        self.cm = connection_manager

    async def upsert_document_metadata(self, doc: Document) -> None:
        await self.cm.execute(
            UPSERT_METADATA_SQL,
            doc.id,
            doc.title,
            doc.url,
            doc.source,
            doc.last_modified_date,
            doc.source_refs or None,
        )
        log.debug(f"Upserted metadata for {doc.title}")

    async def get_last_indexed_at(self, document_id: str) -> datetime | None:
        """Return last indexed timestamp for a document, if present."""
        return await self.cm.fetchval(
            "SELECT last_indexed_at FROM document_metadata WHERE document_id = $1",
            document_id,
        )

    async def add_chunks(self, chunks: list[EmbeddedDocumentChunk]) -> None:
        if not chunks:
            return

        batch = []
        for c in chunks:
            vec_text = "[" + ",".join(str(v) for v in c.embedding) + "]"
            meta_payload = {
                **c.chunk.metadata,
                "embedding_model": c.embedding_model,
                "retrieval_corpus": c.chunk.document.retrieval_corpus,
            }
            meta = json.dumps(meta_payload)
            batch.append((
                c.chunk.chunk_id,                           # $1 chunk_id
                c.chunk.text.replace("\x00", ""),           # $2 text
                c.chunk.document.source_code or "",         # $3 source_code
                c.chunk.document.id,                        # $4 document_id
                c.chunk.document.title.strip(),             # $5 document_title
                vec_text,                                   # $6 embedding_3072
                meta,                                       # $7 metadata
                c.chunk.document.source,                    # $8 source
            ))

        await self.cm.executemany(INSERT_CHUNK_SQL, batch)
        log.info(f"Inserted {len(batch)} chunks")

    async def delete_document_chunks(self, document_id: str) -> None:
        await self.cm.execute(
            "DELETE FROM document_embeddings WHERE document_id = $1", document_id
        )

    async def replace_document(self, doc: Document, chunks: list[EmbeddedDocumentChunk]) -> None:
        """Replace one document's indexed content: delete old chunks, then upsert and insert."""
        await self.delete_document_chunks(doc.id)
        await self.upsert_document_metadata(doc)
        await self.add_chunks(chunks)
