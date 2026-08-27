"""Set-based retrieval metrics for document-level evaluation."""

from __future__ import annotations

from math import log2


class MetricsCalculator:
    """Computes retrieval metrics at K."""

    @staticmethod
    def unique_doc_ids_in_order(doc_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for doc_id in doc_ids:
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc_id)
        return unique

    @staticmethod
    def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
        if not relevant_doc_ids or k <= 0:
            return 0.0
        top_k = set(retrieved_doc_ids[:k])
        return len(top_k & relevant_doc_ids) / len(relevant_doc_ids)

    @staticmethod
    def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top_k = set(retrieved_doc_ids[:k])
        return len(top_k & relevant_doc_ids) / k

    @staticmethod
    def f1_score(recall: float, precision: float) -> float:
        if recall <= 0.0 or precision <= 0.0:
            return 0.0
        return 2.0 * recall * precision / (recall + precision)

    @staticmethod
    def hit_count_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> int:
        top_k = set(retrieved_doc_ids[:k])
        return len(top_k & relevant_doc_ids)

    @staticmethod
    def mrr_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
        if k <= 0 or not relevant_doc_ids:
            return 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
        if k <= 0 or not relevant_doc_ids:
            return 0.0
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            if doc_id in relevant_doc_ids:
                dcg += 1.0 / log2(rank + 1)
        ideal_hits = min(len(relevant_doc_ids), k)
        if ideal_hits == 0:
            return 0.0
        idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        if idcg <= 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def ndcg_at_k_graded(
        retrieved_doc_ids: list[str],
        relevance_by_doc_id: dict[str, int],
        k: int,
    ) -> float:
        if k <= 0 or not relevance_by_doc_id:
            return 0.0

        def gain(relevance: int) -> float:
            return float((2**relevance) - 1)

        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
            relevance = max(0, int(relevance_by_doc_id.get(doc_id, 0)))
            if relevance > 0:
                dcg += gain(relevance) / log2(rank + 1)

        positive_relevances = sorted(
            (max(0, int(relevance)) for relevance in relevance_by_doc_id.values() if int(relevance) > 0),
            reverse=True,
        )
        if not positive_relevances:
            return 0.0

        idcg = 0.0
        for rank, relevance in enumerate(positive_relevances[:k], start=1):
            idcg += gain(relevance) / log2(rank + 1)
        if idcg <= 0.0:
            return 0.0
        return dcg / idcg
