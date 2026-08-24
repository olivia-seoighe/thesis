"""Reciprocal Rank Fusion (RRF) retrieval strategy.

RRF formula: score(d) = sum_i [ 1 / (K + rank_i(d)) ]
where K=60

"""

from models.models import RetrievedChunk, SearchResponse

RRF_K = 60


def rrf_merge(responses: list[SearchResponse], top_k: int) -> list[RetrievedChunk]:
    """
    Merges ranked SearchResponse objects using Reciprocal Rank Fusion.
    
    Args:
        responses: One SearchResponse per retrieval method (vector, keyword,
                   graph, etc.). Each contains chunks ranked by that method.
        top_k: Maximum number of chunks to return.

    Returns:
        Merged list of RetrievedChunk sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for response in responses:
        for rank, chunk in enumerate(response.chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks_by_id[chunk.chunk_id] = chunk

    # Keep only the highest-scoring chunk per document title to prevent one
    # file occupying multiple top-k slots through different chunks.
    best_by_title: dict[str, str] = {}
    for cid in sorted(scores, key=lambda c: scores[c], reverse=True):
        title = chunks_by_id[cid].document_title
        best_by_title.setdefault(title, cid)

    sorted_ids = sorted(best_by_title.values(), key=lambda cid: scores[cid], reverse=True)
    return [
        chunks_by_id[cid].model_copy(update={"score": scores[cid]})
        for cid in sorted_ids[:top_k]
    ]
