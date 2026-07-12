"""Reciprocal Rank Fusion (RRF) retrieval strategy.

Cormack, Clarke, Buettcher (2009): "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods."

RRF formula: score(d) = sum_i [ 1 / (K + rank_i(d)) ]
where K=60 is the dampening constant from the original paper.

Score ranges (K=60, two result lists):
  - Appears in one list only, ranked 1st:  1/61  ≈ 0.016
  - Appears in both lists, ranked 1st each: 2/61 ≈ 0.033
  - Gap at ~0.025 separates single-list hits from cross-list hits.

Extending to three lists (e.g. adding graph retrieval) requires no changes
to this function — just pass a third SearchResponse to the caller.
"""

from models.models import RetrievedChunk, SearchResponse

RRF_K = 60


def rrf_merge(responses: list[SearchResponse], top_k: int) -> list[RetrievedChunk]:
    """Merge ranked SearchResponse objects using Reciprocal Rank Fusion.

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

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        chunks_by_id[cid].model_copy(update={"score": scores[cid]})
        for cid in sorted_ids[:top_k]
    ]
