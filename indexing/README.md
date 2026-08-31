
1. Build two corpora into the same index (separate ingest passes)

INDEXING_RETRIEVAL_CORPUS=summaries docker compose --profile ingest up --build indexing

GRAPH_INDEXING_ENABLED=false INDEXING_RETRIEVAL_CORPUS=code docker compose --profile ingest up --build indexing


2. Run a graph-only summaries pass:

GRAPH_INDEXING_ENABLED=true GRAPH_ONLY=true INDEXING_RETRIEVAL_CORPUS=summaries docker compose --profile ingest up --build indexing

This will:
1. Recompute and replace graph evidence per file.
2. Point evidence back to summaries-mode document IDs.
3. Avoid re-indexing vectors (so no extra summary-vector duplicates).

If you previously set graph off anywhere, force it on explicitly with:
`GRAPH_INDEXING_ENABLED=true GRAPH_ONLY=true INDEXING_RETRIEVAL_CORPUS=summaries docker compose --profile ingest up --build indexing`


3.  Run evaluation for summaries corpus

python3 evaluation/scripts/run_retrieval_eval.py \
  --retrieval-url http://localhost:18000 \
  --dataset-dir evaluation/datasets/v1 \
  --strategies keyword-bm25,vector,hybrid-bm25,keyword-bm25-service-aware,vector-service-aware,hybrid-bm25-service-aware \
  --k-values 10,15 \
  --retrieval-corpus summaries

4. Run evaluation for code corpus

python3 evaluation/scripts/run_retrieval_eval.py \
  --retrieval-url http://localhost:18000 \
  --dataset-dir evaluation/datasets/v1 \
  --strategies keyword-bm25,vector,hybrid-bm25,keyword-bm25-service-aware,vector-service-aware,hybrid-bm25-service-aware \
  --k-values 10,15 \
  --retrieval-corpus code

