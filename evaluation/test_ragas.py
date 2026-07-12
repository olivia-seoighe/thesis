"""RAGAS evaluation of the RAG pipeline against sample golden queries.

Metrics (all reference-free — no ground_truth required):
  - faithfulness:       Are claims in the answer supported by retrieved context?
  - answer_relevancy:   Is the answer relevant to the question?
  - context_precision:  Is the top-ranked context relevant?

Run:
    pytest evaluation/test_ragas.py -v -s

Required env vars (in .env):
    PGHOST, PGDATABASE, PGUSER, PGPASSWORD
    OPENAI_API_KEY, OPENAI_BASE_URL
    OPENAI_MODEL         (default: claude-sonnet-4-6)
    EVAL_SOURCE          (default: service_A)
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

GOLDEN_QUERIES_PATH = Path(__file__).parent / "sample_golden_queries.json"
FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70

REQUIRED_VARS = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "OPENAI_API_KEY"]


def _check_env() -> None:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        pytest.skip(f"Missing env vars: {missing}")


def load_golden_queries() -> list[dict]:
    with open(GOLDEN_QUERIES_PATH) as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_ragas_pipeline() -> None:
    """End-to-end RAGAS evaluation: retrieval → generation → faithfulness & relevancy."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        pytest.skip("ragas not installed — run: pip install ragas datasets")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        pytest.skip("langchain-openai not installed — run: pip install langchain-openai")

    _check_env()

    from clients.search_client import SearchClient
    from llm_client import LLMClient
    from models import SearchRequest

    source = os.getenv("EVAL_SOURCE", "service_A")
    llm_model = os.getenv("OPENAI_MODEL", "claude-sonnet-4-6")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.environ["OPENAI_API_KEY"]

    queries = load_golden_queries()
    search_client = SearchClient()
    llm_client = LLMClient()

    records: list[dict] = []

    try:
        for item in queries:
            question = item["input"]
            t0 = time.time()

            # Retrieve
            req = SearchRequest(query=question, sources=[source], top_k=5)
            resp = await search_client.search(req)
            contexts = [c.text for c in resp.chunks]

            # Generate
            chunks_for_llm = [
                {
                    "document_title": c.document_title,
                    "url": c.url,
                    "text": c.text,
                    "score": c.score,
                }
                for c in resp.chunks
            ]
            answer, _ = await llm_client.generate(
                query=question,
                chunks=chunks_for_llm,
                model=llm_model,
            )

            latency_ms = (time.time() - t0) * 1000
            records.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "latency_ms": latency_ms,
            })

            print(f"  [{len(records)}/{len(queries)}] {question[:70]}…  ({latency_ms:.0f}ms)")

    finally:
        await search_client.close()
        await llm_client.close()

    # ── Latency summary ──────────────────────────────────────────────────────
    latencies = [r["latency_ms"] for r in records]
    print(f"\nLatency — mean: {sum(latencies)/len(latencies):.0f}ms  "
          f"max: {max(latencies):.0f}ms  min: {min(latencies):.0f}ms")

    # ── RAGAS evaluation ─────────────────────────────────────────────────────
    ragas_llm = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
    )
    ragas_embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=api_key,
        base_url=base_url or "https://api.openai.com/v1",
    )

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
        }
        for r in records
    ])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = result.to_pandas()
    print("\n" + "=" * 60)
    print("RAGAS SCORES")
    print("=" * 60)
    print(scores[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string(index=False))
    print("=" * 60)
    print(f"\nMean faithfulness:       {scores['faithfulness'].mean():.3f}  (threshold {FAITHFULNESS_THRESHOLD})")
    print(f"Mean answer_relevancy:   {scores['answer_relevancy'].mean():.3f}  (threshold {ANSWER_RELEVANCY_THRESHOLD})")
    print(f"Mean context_precision:  {scores['context_precision'].mean():.3f}  (threshold {CONTEXT_PRECISION_THRESHOLD})")

    assert scores["faithfulness"].mean() >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {scores['faithfulness'].mean():.3f} < {FAITHFULNESS_THRESHOLD}"
    )
    assert scores["answer_relevancy"].mean() >= ANSWER_RELEVANCY_THRESHOLD, (
        f"Answer relevancy {scores['answer_relevancy'].mean():.3f} < {ANSWER_RELEVANCY_THRESHOLD}"
    )
    assert scores["context_precision"].mean() >= CONTEXT_PRECISION_THRESHOLD, (
        f"Context precision {scores['context_precision'].mean():.3f} < {CONTEXT_PRECISION_THRESHOLD}"
    )


@pytest.mark.asyncio
async def test_retrieval_latency() -> None:
    """Assert P95 end-to-end retrieval latency is below 3 seconds."""
    _check_env()

    sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
    from clients.search_client import SearchClient
    from models import SearchRequest

    source = os.getenv("EVAL_SOURCE", "service_A")
    queries = load_golden_queries()
    search_client = SearchClient()
    latencies: list[float] = []

    try:
        for item in queries:
            t0 = time.time()
            await search_client.search(SearchRequest(query=item["input"], sources=[source], top_k=5))
            latencies.append((time.time() - t0) * 1000)
    finally:
        await search_client.close()

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"\nRetrieval latency — P50: {p50:.0f}ms  P95: {p95:.0f}ms")
    assert p95 < 3000, f"P95 retrieval latency {p95:.0f}ms exceeds 3000ms SLA"
