"""Verify structured-query routing against the curated aggregate queries.

Run: python evaluation/scripts/verify_structured_router.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from retrieval.clients.search_client import SearchClient
from retrieval.graph.structured_router import maybe_route_structured_query
from retrieval.models.models import SearchRequest
from retrieval.strategies.metadata_aware import load_service_catalogue

DSN = dict(host="localhost", port=15433, user="raguser", password="ragpassword", database="ragdb")
KNOWN_QUERY_IDS = [
    "GQ016", "GQ017", "GQ025", "GQ027", "GQ029", "GQ059", "GQ060", "GQ061",
    "GQ062", "GQ063", "GQ064", "GQ065", "GQ067", "GQ087", "GQ088", "GQ089",
]


def parse_gold_source_files(raw) -> set[tuple[str, str]]:
    if isinstance(raw, list):
        return {(i.get("service", "").strip(), i.get("file", "").strip()) for i in raw if isinstance(i, dict)}
    if isinstance(raw, str):
        parts = raw.split(";") if ";" in raw and "\n" not in raw else raw.splitlines()
        pairs = set()
        for p in parts:
            p = p.strip()
            if "::" in p:
                s, f = p.split("::", 1)
                pairs.add((s.strip(), f.strip()))
        return pairs
    return set()


def load_service_aliases() -> list[tuple[str, list[str]]]:
    catalogue_path = ROOT / "service_acronyms.json"
    if not catalogue_path.exists():
        return []
    entries = load_service_catalogue(catalogue_path)
    return [(entry.source, sorted({*entry.short_forms, *entry.long_forms})) for entry in entries]


async def main() -> None:
    with open(ROOT / "evaluation" / "golden_queries2.json") as f:
        data = json.load(f)
    by_id = {q["master_query_id"]: q for q in data["queries"]}
    service_aliases = load_service_aliases()

    search_client = SearchClient(**DSN)
    results = []
    try:
        for qid in KNOWN_QUERY_IDS:
            q = by_id[qid]
            query_text = q.get("query") or q.get("question") or ""
            gold = parse_gold_source_files(q["gold_source_files"])

            request = SearchRequest(query=query_text, top_k=200, retrieval_corpus="code")
            response = await maybe_route_structured_query(
                request, search_client=search_client, service_aliases=service_aliases, start_time=0.0,
            )

            if response is None:
                print(f"{qid}: DID NOT FIRE  (query: {query_text[:80]!r})")
                results.append((qid, 0.0, 0.0, 0.0))
                continue

            predicted = {(c.source, c.document_title) for c in response.chunks}
            tp = len(predicted & gold)
            recall = tp / len(gold) if gold else 0.0
            precision = tp / len(predicted) if predicted else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            results.append((qid, recall, precision, f1))

            route_debug = response.chunks[0].metadata.get("structured_route", {}) if response.chunks else {}
            print(
                f"{qid}: recall={recall:.3f} precision={precision:.3f} f1={f1:.3f} "
                f"(gold={len(gold)}, predicted={len(predicted)}, tp={tp}) "
                f"template={route_debug.get('template_id')} predicate={route_debug.get('predicate')} "
                f"shape={route_debug.get('routing_shape')} sim={route_debug.get('similarity', 0):.3f}"
            )
            missing = gold - predicted
            if missing:
                print(f"    missing: {sorted(missing)}")

        n = len(results)
        mean_recall = sum(r[1] for r in results) / n
        mean_precision = sum(r[2] for r in results) / n
        mean_f1 = sum(r[3] for r in results) / n
        print(f"\nMEAN over {n} queries: recall={mean_recall:.3f} precision={mean_precision:.3f} f1={mean_f1:.3f}")
    finally:
        await search_client.close()


if __name__ == "__main__":
    asyncio.run(main())
