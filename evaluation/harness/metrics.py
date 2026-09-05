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
    def recall_ceiling_at_k(relevant_doc_count: int, k: int) -> float:
        if relevant_doc_count <= 0 or k <= 0:
            return 0.0
        return min(k, relevant_doc_count) / relevant_doc_count

    @staticmethod
    def ceiling_adjusted_recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
        ceiling = MetricsCalculator.recall_ceiling_at_k(len(relevant_doc_ids), k)
        if ceiling <= 0.0:
            return 0.0
        return MetricsCalculator.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k) / ceiling

    @staticmethod
    def evidence_group_recall_at_k(
        retrieved_doc_ids: list[str],
        relevant_doc_to_group: dict[str, str],
        k: int,
    ) -> float:
        gold_groups = {group for group in relevant_doc_to_group.values() if group}
        if not gold_groups or k <= 0:
            return 0.0
        retrieved_groups = {
            relevant_doc_to_group[doc_id]
            for doc_id in retrieved_doc_ids[:k]
            if doc_id in relevant_doc_to_group and relevant_doc_to_group[doc_id]
        }
        return len(retrieved_groups & gold_groups) / len(gold_groups)

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
    def evidence_group_hit_count_at_k(
        retrieved_doc_ids: list[str],
        relevant_doc_to_group: dict[str, str],
        k: int,
    ) -> int:
        return len(
            {
                relevant_doc_to_group[doc_id]
                for doc_id in retrieved_doc_ids[:k]
                if doc_id in relevant_doc_to_group and relevant_doc_to_group[doc_id]
            }
        )

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
