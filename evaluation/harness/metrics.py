"""Set-based retrieval metrics for document-level evaluation."""

from __future__ import annotations


class MetricsCalculator:
    """Computes recall, precision, and F1 at K."""

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
