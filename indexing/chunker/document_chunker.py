import hashlib
from typing import List

from indexing.models.document import Document, DocumentChunk


class DocumentChunker:
    def __init__(self, chunk_size: int = 1500, overlap_ratio: float = 0.15):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)

    def chunk(self, doc: Document) -> List[DocumentChunk]:
        words = doc.text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        windows: List[List[str]] = []
        i = 0
        while i < len(words):
            windows.append(words[i : i + self.chunk_size])
            i += step

        total = len(windows)
        chunks = []
        for idx, window in enumerate(windows, start=1):
            content = " ".join(window)
            prefix = (
                f"TITLE: {doc.title} | SOURCE: {doc.source} | CORPUS: {doc.retrieval_corpus} "
                f"| PART: {idx} of {total} | CONTENT: "
            )
            text = prefix + content

            chunk_id = hashlib.sha256(f"{doc.id}::{idx}".encode()).hexdigest()

            metadata = {
                "chunk_index": idx,
                "total_chunks": total,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "document_url": doc.url,
                "retrieval_corpus": doc.retrieval_corpus,
            }

            chunks.append(
                DocumentChunk(chunk_id=chunk_id, text=text, document=doc, metadata=metadata)
            )

        return chunks