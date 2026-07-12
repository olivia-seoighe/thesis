"""Retrieval evaluation for the code RAG pipeline using sample golden queries.

Pattern:
  - loads golden queries from a local JSON file
  - runs vector search for each query
  - uses Arize Phoenix llm_classify to label each (query, context) pair
  - asserts relevance_rate >= RELEVANCE_THRESHOLD

Run:
    cd <repo-root>
    pytest evals/test_retrieval.py -v -s

Required env vars (in .env):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    OPENAI_API_KEY
    EMBEDDING_MODEL   (default: openai/text-embedding-3-large)
    EVAL_SOURCE       (default: service_A)
    OPENAI_MODEL      (default: gpt-4o — used by Phoenix for classification)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv
from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)

# Allow imports from retrieval/ without installing as a package
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "retrieval"))

from clients.search_client import SearchClient  # noqa: E402
from models.models import SearchRequest  # noqa: E402

GOLDEN_QUERIES_PATH = Path(__file__).parent / "sample_golden_queries.json"
RELEVANCE_THRESHOLD = 0.95


def load_golden_queries() -> list[str]:
    with open(GOLDEN_QUERIES_PATH) as f:
        data = json.load(f)
    return [item["input"] for item in data]


@pytest.mark.asyncio
async def test_retrieval_relevance() -> None:
    """Assert that vector search returns relevant chunks for all golden queries."""
    load_dotenv()

    required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "OPENAI_API_KEY"]
    for var in required:
        if not os.getenv(var):
            pytest.skip(f"Required environment variable {var} not set")

    source = os.getenv("EVAL_SOURCE", "service_A")
    model_name = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    queries = load_golden_queries()

    search_client = SearchClient()

    evaluation_dict: dict[str, list] = {
        "input": [],
        "reference": [],
        "retrieved_titles": [],
    }

    try:
        for query in queries:
            search_request = SearchRequest(query=query, sources=[source], top_k=5)
            search_response = await search_client.search(search_request)

            contexts = [chunk.text for chunk in search_response.chunks]
            titles = [chunk.document_title for chunk in search_response.chunks]

            evaluation_dict["input"].append(query)
            evaluation_dict["reference"].append(contexts)
            evaluation_dict["retrieved_titles"].append(titles)

    finally:
        await search_client.close()

    evaluation_df = pd.DataFrame(evaluation_dict)

    # Arize Phoenix LLM-based relevance classification
    model = OpenAIModel(model=openai_model)
    rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())

    retrieval_eval_df = llm_classify(
        data=evaluation_df,
        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
        model=model,
        rails=rails,
        provide_explanation=True,
    )

    results_df = pd.concat(
        [
            evaluation_df[["input", "retrieved_titles"]],
            retrieval_eval_df[["label", "explanation"]],
        ],
        axis=1,
    )
    results_df["reference"] = evaluation_df["reference"]

    # --- Print results ---
    passes = results_df[results_df["label"] == "relevant"]
    failures = results_df[results_df["label"] != "relevant"]

    print(f"\n{'=' * 60}")
    print(
        f"SUMMARY: {len(passes)} PASSED, {len(failures)} FAILED "
        f"out of {len(results_df)} total"
    )
    print(f"{'=' * 60}")

    if not passes.empty:
        print("\n--- PASSED ---")
        for i, row in passes.iterrows():
            print(f"\n[PASS {i}] {row['input']}")
            print(f"  Docs:  {row['retrieved_titles'][:3]}")
            print(f"  Label: {row['label']}")
            print(f"  Why:   {row['explanation'][:200]}...")

    if not failures.empty:
        print("\n--- FAILED ---")
        for i, row in failures.iterrows():
            print(f"\n[FAIL {i}] {row['input']}")
            print(f"  Docs:  {row['retrieved_titles']}")
            print(f"  Label: {row['label']}")
            print(f"  Why:   {row['explanation']}")
            contexts = row["reference"]
            if contexts:
                print(f"  Top context: {str(contexts[0])[:300]}...")
            else:
                print("  Top context: [EMPTY — nothing retrieved]")

    relevance_rate = (results_df["label"] == "relevant").sum() / len(results_df)
    print(f"\nRelevance Rate: {relevance_rate:.1%} (threshold: {RELEVANCE_THRESHOLD:.0%})")

    assert relevance_rate >= RELEVANCE_THRESHOLD, (
        f"Relevance rate {relevance_rate:.1%} is below threshold {RELEVANCE_THRESHOLD:.0%}. "
        f"See failures above."
    )