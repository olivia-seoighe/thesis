
1. Build two corpora into the same index (separate ingest passes)

INDEXING_RETRIEVAL_CORPUS=summaries docker compose --profile ingest up --build indexing
INDEXING_RETRIEVAL_CORPUS=code docker compose --profile ingest up --build indexing


2. Run evaluation for summaries corpus

python3 evaluation/scripts/run_retrieval_eval.py \
  --retrieval-url http://localhost:18000 \
  --dataset-dir evaluation/datasets/v1 \
  --strategies keyword-bm25,vector,hybrid-bm25,keyword-bm25-service-aware,vector-service-aware,hybrid-bm25-service-aware \
  --k-values 10,15 \
  --retrieval-corpus summaries

3. Run evaluation for code corpus

python3 evaluation/scripts/run_retrieval_eval.py \
  --retrieval-url http://localhost:18000 \
  --dataset-dir evaluation/datasets/v1 \
  --strategies keyword-bm25,vector,hybrid-bm25,keyword-bm25-service-aware,vector-service-aware,hybrid-bm25-service-aware \
  --k-values 10,15 \
  --retrieval-corpus code

