"""Heuristic metrics for end-to-end generation evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_ABSTAIN_HINTS = (
    "cannot determine",
    "can't determine",
    "not identified",
    "not enough",
    "insufficient",
    "do not identify",
    "would need additional",
)


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if len(token) > 2}


@dataclass(frozen=True)
class GenerationMetricScores:
    correctness_score: float
    completeness_score: float
    abstention_quality_score: float
    hallucination_flag: int
    hallucination_notes: str
    faithfulness_score: float
    score_confidence: str
    judge_rationale: str


class GenerationMetricsCalculator:
    """Computes low-cost heuristic quality metrics for generated answers."""

    @staticmethod
    def score_answer(
        *,
        generated_answer: str,
        gold_answer: str,
        answerable: bool,
        citations_count: int,
    ) -> GenerationMetricScores:
        answer_tokens = _tokenize(generated_answer)
        gold_tokens = _tokenize(gold_answer)
        overlap = answer_tokens & gold_tokens

        completeness = len(overlap) / len(gold_tokens) if gold_tokens else 0.0
        correctness = len(overlap) / len(answer_tokens) if answer_tokens else 0.0
        abstention = GenerationMetricsCalculator._abstention_score(generated_answer, answerable=answerable)

        has_citations = citations_count > 0
        hallucination_flag = int(bool(generated_answer.strip()) and not has_citations and answerable)
        hallucination_notes = "No citations returned for answerable query." if hallucination_flag else ""
        faithfulness = 1.0 if has_citations else (1.0 if not answerable else 0.0)

        confidence = "high" if len(gold_tokens) >= 25 else ("medium" if len(gold_tokens) >= 10 else "low")
        rationale = (
            f"token_overlap={len(overlap)} gold_tokens={len(gold_tokens)} "
            f"answer_tokens={len(answer_tokens)} citations={citations_count}"
        )

        return GenerationMetricScores(
            correctness_score=correctness,
            completeness_score=completeness,
            abstention_quality_score=abstention,
            hallucination_flag=hallucination_flag,
            hallucination_notes=hallucination_notes,
            faithfulness_score=faithfulness,
            score_confidence=confidence,
            judge_rationale=rationale,
        )

    @staticmethod
    def citation_precision(*, cited_files: set[str], cited_services: set[str], gold_files: set[str], gold_services: set[str]) -> float:
        if not cited_files and not cited_services:
            return 0.0
        file_hits = len(cited_files & gold_files)
        service_hits = len(cited_services & gold_services)
        denom = len(cited_files) + len(cited_services)
        return (file_hits + service_hits) / denom if denom else 0.0

    @staticmethod
    def overlap_ratio(found: set[str], expected: set[str]) -> float:
        if not expected:
            return 0.0
        return len(found & expected) / len(expected)

    @staticmethod
    def _abstention_score(answer_text: str, *, answerable: bool) -> float:
        lowered = (answer_text or "").lower()
        abstained = any(hint in lowered for hint in _ABSTAIN_HINTS)
        if answerable:
            return 0.0 if abstained else 1.0
        return 1.0 if abstained else 0.0
