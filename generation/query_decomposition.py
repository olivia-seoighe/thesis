from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SubQuery:
    id: str
    query: str


@dataclass(frozen=True)
class DecompositionPlan:
    should_decompose: bool
    reason: str
    sub_queries: tuple[SubQuery, ...] = ()


_TOPIC_RE = re.compile(r"\b([a-z0-9]+(?:[_-][a-z0-9]+)+)\b", re.IGNORECASE)


def build_decomposition_plan(query: str) -> DecompositionPlan:
    text = " ".join((query or "").split())
    lowered = text.lower()
    if not text:
        return DecompositionPlan(False, "empty query")

    asks_multiple = text.count("?") > 1 or ", and which " in lowered or " and which " in lowered
    if not asks_multiple:
        return DecompositionPlan(False, "single ask query")

    if ("produc" in lowered and "consum" in lowered) and "topic" in lowered:
        topic = _extract_topic_token(text)
        producer_q = f"Which service produces {topic}?" if topic else "Which service produces this topic?"
        consumer_q = f"Which services consume {topic}?" if topic else "Which services consume this topic?"
        return DecompositionPlan(
            True,
            "Detected producer+consumer multi-ask topology query.",
            sub_queries=(SubQuery("producer", producer_q), SubQuery("consumer", consumer_q)),
        )

    return DecompositionPlan(False, "multi-ask pattern not confidently decomposable")


def _extract_topic_token(query: str) -> str:
    for match in _TOPIC_RE.finditer(query):
        token = match.group(1)
        if "_" in token or "-" in token:
            return token
    return ""
